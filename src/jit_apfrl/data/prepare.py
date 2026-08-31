from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ALIBABA_COMMIT = "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71"
SOURCES = {
    "nasa_cmapss": {
        "url": "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
        "filename": "CMAPSSData.zip",
    },
    "alibaba_pods": {
        "url": (
            "https://raw.githubusercontent.com/alibaba/clusterdata/"
            f"{ALIBABA_COMMIT}/cluster-trace-gpu-v2023/csv/openb_pod_list_default.csv"
        ),
        "filename": "openb_pod_list_default.csv",
    },
    "alibaba_nodes": {
        "url": (
            "https://raw.githubusercontent.com/alibaba/clusterdata/"
            f"{ALIBABA_COMMIT}/cluster-trace-gpu-v2023/csv/openb_node_list_all_node.csv"
        ),
        "filename": "openb_node_list_all_node.csv",
    },
}

SENSOR_NUMBERS = (2, 3, 4, 7, 8, 9, 11, 12)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "jit-apfrl/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def ensure_raw_data(raw_dir: Path, allow_download: bool) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for key, source in SOURCES.items():
        path = raw_dir / source["filename"]
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError(f"required public dataset is missing: {path}")
            _download(str(source["url"]), path)
        if path.stat().st_size == 0:
            raise ValueError(f"downloaded file is empty: {path}")
        manifest[key] = {
            **source,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return manifest


def _load_fd001(zip_path: Path) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("train_FD001.txt")]
        if len(candidates) != 1:
            raise ValueError("CMAPSS archive does not contain exactly one train_FD001.txt")
        with archive.open(candidates[0]) as stream:
            data = np.loadtxt(stream, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 26:
        raise ValueError(f"unexpected FD001 schema: {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("FD001 contains non-finite data")
    return data


def _fit_minmax(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = train.min(axis=0)
    high = train.max(axis=0)
    return low, np.maximum(high - low, 1e-12)


def _process_fd001(data: np.ndarray) -> dict[str, np.ndarray]:
    engine_ids = data[:, 0].astype(np.int64)
    cycles = data[:, 1]
    sensor_columns = [5 + sensor_number - 1 for sensor_number in SENSOR_NUMBERS]
    selected = data[:, sensor_columns]
    train_mask = engine_ids <= 80
    eval_mask = engine_ids > 80
    if np.any(np.isin(np.unique(engine_ids[train_mask]), np.unique(engine_ids[eval_mask]))):
        raise AssertionError("engine-level split overlap")
    low, span = _fit_minmax(selected[train_mask])
    normalized = np.clip((selected - low) / span, 0.0, 1.0)
    maximum_cycle = {engine: cycles[engine_ids == engine].max() for engine in np.unique(engine_ids)}
    health = np.asarray(
        [max(0.0, 1.0 - cycle / maximum_cycle[int(engine)]) for engine, cycle in zip(engine_ids, cycles, strict=True)],
        dtype=np.float64,
    )
    return {
        "train_equipment": normalized[train_mask],
        "train_health": health[train_mask],
        "train_engine_id": engine_ids[train_mask],
        "train_cycle": cycles[train_mask].astype(np.int64),
        "eval_equipment": normalized[eval_mask],
        "eval_health": health[eval_mask],
        "eval_engine_id": engine_ids[eval_mask],
        "eval_cycle": cycles[eval_mask].astype(np.int64),
        "feature_min": low,
        "feature_span": span,
    }


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _process_alibaba(path: Path) -> dict[str, np.ndarray]:
    raw_rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "cpu_milli",
            "memory_mib",
            "num_gpu",
            "gpu_milli",
            "qos",
            "creation_time",
            "scheduled_time",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"unexpected Alibaba pod schema: {reader.fieldnames}")
        for row in reader:
            wait_seconds = max(0.0, _number(row, "scheduled_time") - _number(row, "creation_time"))
            qos = (row.get("qos") or "").strip().lower()
            qos_pressure = {"ls": 1.0, "burstable": 0.6, "be": 0.3}.get(qos, 0.5)
            raw_rows.append(
                [
                    max(0.0, _number(row, "cpu_milli")),
                    max(0.0, _number(row, "memory_mib")),
                    max(0.0, _number(row, "num_gpu") * _number(row, "gpu_milli") / 1000.0),
                    wait_seconds,
                    qos_pressure,
                ]
            )
    raw = np.asarray(raw_rows, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 100 or raw.shape[1] != 5:
        raise ValueError(f"unexpected Alibaba row matrix: {raw.shape}")
    split = int(0.8 * len(raw))
    if split <= 0 or split >= len(raw):
        raise ValueError("invalid chronological workload split")
    train_raw = raw[:split]
    scales = np.maximum(np.percentile(train_raw[:, :4], 99, axis=0), 1.0)

    def transform(rows: np.ndarray) -> np.ndarray:
        resources = np.clip(rows[:, :3] / scales[:3], 0.0, 1.0)
        wait_pressure = np.clip(np.log1p(rows[:, 3]) / np.log1p(scales[3]), 0.0, 1.0)
        pressure = np.clip(0.6 * rows[:, 4] + 0.4 * wait_pressure, 0.0, 1.0)
        output = np.column_stack([resources, pressure])
        if not np.all(np.isfinite(output)):
            raise ValueError("non-finite Alibaba preprocessing output")
        return output

    return {
        "train_workloads": transform(raw[:split]),
        "eval_workloads": transform(raw[split:]),
        "workload_p99": scales,
        "workload_split_row": np.asarray(split, dtype=np.int64),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def prepare_datasets(
    project_root: str | Path,
    *,
    allow_download: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    manifest_dir = root / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest = ensure_raw_data(raw_dir, allow_download=allow_download)

    fd001 = _process_fd001(_load_fd001(raw_dir / SOURCES["nasa_cmapss"]["filename"]))
    alibaba = _process_alibaba(raw_dir / SOURCES["alibaba_pods"]["filename"])
    arrays = {**fd001, **alibaba}
    benchmark_path = processed_dir / "benchmark_v1.npz"
    _write_npz(benchmark_path, arrays)

    raw_manifest_path = manifest_dir / "sources.json"
    raw_manifest_path.write_text(json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = {
        str(benchmark_path.relative_to(root)): sha256_file(benchmark_path),
    }
    (manifest_dir / "processed_checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "raw_manifest": str(raw_manifest_path),
        "benchmark": str(benchmark_path),
        "checksums": checksums,
        "counts": {
            "train_equipment": int(len(arrays["train_equipment"])),
            "eval_equipment": int(len(arrays["eval_equipment"])),
            "train_workloads": int(len(arrays["train_workloads"])),
            "eval_workloads": int(len(arrays["eval_workloads"])),
        },
    }
