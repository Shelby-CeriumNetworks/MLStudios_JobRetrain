

import json, os, hashlib, time
from pathlib import Path

def now_ts():
    return int(time.time())

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_json(path: str, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def append_jsonl(path: str, record: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()