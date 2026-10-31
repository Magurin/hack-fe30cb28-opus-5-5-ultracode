"""Soup's tokenization of text SFT rows without `datasets.map` (organizers' answer
t.me/souptasterschat/2989 to /2987 p.4: our own batch tokenization in memory is allowed, with the
same input_ids and labels).

Soup's `_map_text_sft_rows` runs `Dataset.from_list(rows).map(checked_format_row,
with_indices=True, remove_columns=["messages"])`. On a dataset in memory `map` writes no cache
file, but to fingerprint its result it hashes `checked_format_row` with dill, and that closure holds
the tokenizer: 2.0 s on a T4 machine (2.1 s on Kaggle) of the map's 2.6 s, for 256 rows that take
0.45 s to format. Here: the same `format_row` and the same check (`ensure_causal_loss_target`, the
same ValueError naming the row) on the same rows, read back from the same Arrow table as `map`
would pass them, in the same order, then `Dataset.from_list` of the results: the same columns,
features and values (solution/test_dataprep.py). Soup's own hashes of the data (the rewind log's
`dataset_fingerprint`, the experiment's training data hash) are taken from the rows and the file,
not from this dataset, so they do not change. Anything unusual (a column besides `messages`, rows
whose results differ in keys, a column that is not a list of integers, an empty split) goes to
Soup's own `map`.
"""
import functools


def _formatted(rows, format_row, split, max_length):
    """The results of `format_row` for every row, or None where Soup's own map should run."""
    from datasets import Dataset

    from soup_cli.data.loss_mask import NoCausalLossTargetError, ensure_causal_loss_target

    if not rows:
        return None
    source = Dataset.from_list(rows)
    if source.column_names != ["messages"]:
        return None
    results, keys = [], None
    for row_index, example in enumerate(source):
        try:
            formatted = format_row(example)
            labels = formatted.get("labels")
            if labels is not None:
                ensure_causal_loss_target(labels, max_length=max_length)
        except NoCausalLossTargetError as exc:
            raise ValueError(f"{split} row {row_index + 1}: {exc}") from exc
        if keys is None:
            keys = list(formatted)
            if "messages" in keys:
                return None
        elif list(formatted) != keys:
            return None
        results.append(formatted)
    return results


def _all_int_lists(dataset):
    import pyarrow as pa

    return all(field.type == pa.list_(pa.int64()) for field in dataset.data.schema)


def patch_tokenization():
    from soup_cli.trainer import sft

    original = sft._map_text_sft_rows
    if getattr(original, "_soup_kernels_patched", False):
        return

    @functools.wraps(original)
    def map_text_sft_rows(rows, *, format_row, split, max_length):
        from datasets import Dataset

        results = _formatted(rows, format_row, split, max_length)
        if results is not None:
            dataset = Dataset.from_list(results)
            if _all_int_lists(dataset):
                return dataset
        return original(rows, format_row=format_row, split=split, max_length=max_length)

    map_text_sft_rows._soup_kernels_patched = True
    map_text_sft_rows._soup_original = original
    sft._map_text_sft_rows = map_text_sft_rows
