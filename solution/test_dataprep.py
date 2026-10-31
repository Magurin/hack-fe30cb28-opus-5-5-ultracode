"""Our tokenization of text SFT rows (solution/dataprep.py) gives the dataset Soup's map gives:
the same columns, features and values, the same errors; on the task's own data and tokenizer when
they are here (SOLUTION_TEST_DATA: more train files, separated by os.pathsep)."""
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _soup_map():
    from soup_cli.trainer import sft

    return getattr(sft._map_text_sft_rows, "_soup_original", sft._map_text_sft_rows)


def _no_soup_map(*args, **kwargs):
    raise AssertionError("went to Soup's map")


def _ours(fallback=True):
    """Our function; with fallback=False Soup's map behind it fails the test."""
    from soup_cli.trainer import sft

    from solution import dataprep

    saved = sft._map_text_sft_rows
    try:
        sft._map_text_sft_rows = _soup_map() if fallback else _no_soup_map
        dataprep.patch_tokenization()
        return sft._map_text_sft_rows
    finally:
        sft._map_text_sft_rows = saved


def _same(a, b):
    assert a.column_names == b.column_names
    assert a.features == b.features
    assert a.data.schema == b.data.schema
    assert len(a) == len(b)
    for column in a.column_names:
        assert a[column] == b[column], column


def _toy_format_row(example):
    """Characters as token ids; the assistant's characters are the labels, as in Soup."""
    ids, labels = [], []
    for message in example["messages"]:
        tokens = [ord(c) % 1000 for c in f"<{message['role']}>{message['content']}"][:40]
        ids += tokens
        labels += tokens if message["role"] == "assistant" else [-100] * len(tokens)
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def _rows(n):
    return [{"messages": [{"role": "user", "content": f"question {i}" * (i % 5 + 1)},
                          {"role": "assistant", "content": f"answer {i}" * (i % 3 + 1)}]}
            for i in range(n)]


@pytest.mark.parametrize("n", [1, 7, 1500])
def test_same_dataset_as_soup(n):
    from solution import dataprep

    rows = _rows(n)
    kwargs = dict(format_row=_toy_format_row, split="train", max_length=512)
    assert dataprep._formatted(rows, **kwargs) is not None
    _same(_ours(fallback=False)(rows, **kwargs), _soup_map()(rows, **kwargs))


def test_same_error_as_soup():
    rows = _rows(5)
    rows[3]["messages"][1]["content"] = ""  # no causal-loss target left
    rows[3]["messages"][1]["role"] = "user"
    errors = []
    for run in (_ours(fallback=False), _soup_map()):
        with pytest.raises(ValueError) as info:
            run(rows, format_row=_toy_format_row, split="validation", max_length=512)
        errors.append((str(info.value), type(info.value.__cause__)))
    assert errors[0] == errors[1]
    assert errors[0][0].startswith("validation row 4: ")


def test_unusual_rows_go_to_soup():
    rows = [dict(row, source=f"s{i}") for i, row in enumerate(_rows(4))]  # a column besides messages
    kwargs = dict(format_row=_toy_format_row, split="train", max_length=512)
    _same(_ours()(rows, **kwargs), _soup_map()(rows, **kwargs))

    def uneven(example):
        out = _toy_format_row(example)
        if len(example["messages"][0]["content"]) > 30:
            out["extra"] = [1]
        return out

    rows = _rows(6)
    outcomes = []
    for run in (_ours(), _soup_map()):
        try:
            outcomes.append(("ok", run(rows, format_row=uneven, split="train", max_length=512)))
        except Exception as exc:  # noqa: BLE001 - the same outcome either way
            outcomes.append(("error", type(exc)))
    assert outcomes[0][0] == outcomes[1][0]
    if outcomes[0][0] == "ok":
        _same(outcomes[0][1], outcomes[1][1])
    else:
        assert outcomes[0][1] == outcomes[1][1]


def _task_files():
    files = [REPO / "data" / "train.jsonl"]
    files += [Path(p) for p in os.environ.get("SOLUTION_TEST_DATA", "").split(os.pathsep) if p]
    return [f for f in files if f.is_file()]


@pytest.mark.parametrize("train_file", _task_files(), ids=lambda f: f.name)
def test_task_data_same_as_soup(train_file, tmp_path):
    import yaml
    from transformers import AutoTokenizer

    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import data_config_for_task, load_dataset, task_preserves_source_columns
    from soup_cli.data.sft_format import build_format_row

    config = yaml.safe_load((REPO / "task" / "task.yaml").read_text(encoding="utf-8"))
    config["data"]["train"] = str(train_file)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cfg = load_config(path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base, local_files_only=True)
    except OSError:
        pytest.skip(f"no local copy of {cfg.base}'s tokenizer")
    rows = load_dataset(data_config_for_task(cfg.data, cfg.task),
                        preserve_source_columns=task_preserves_source_columns(cfg.task))["train"]
    format_row = build_format_row(tokenizer=tokenizer, data_cfg=cfg.data, training_cfg=cfg.training)
    kwargs = dict(format_row=format_row, split="train", max_length=cfg.data.max_length)
    _same(_ours(fallback=False)(rows, **kwargs), _soup_map()(rows, **kwargs))
