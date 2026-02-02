from __future__ import annotations

import json
import os
import shutil
import tempfile
from configparser import ConfigParser, Error
from pathlib import Path
from typing import Dict, Optional


def config_to_dict(config: ConfigParser) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for section in config.sections():
        out[section] = {k: v for k, v in config.items(section)}
    return out


def _backup_path(path: Path) -> Path:
    return Path(str(path) + ".bak")


def atomic_write_ini(path: Path, config: ConfigParser) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=tmp_dir)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            config.write(f)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copy2(path, _backup_path(path))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=tmp_dir)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copy2(path, _backup_path(path))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def load_ini_with_backup(path: Path, defaults: Dict[str, Dict[str, str]]) -> ConfigParser:
    config = ConfigParser()
    config.read_dict(defaults)
    if not path.exists():
        atomic_write_ini(path, config)
        return config
    try:
        config.read(path, encoding="utf-8")
        return config
    except Error:
        backup = _backup_path(path)
        if backup.exists():
            shutil.copy2(backup, path)
            config.read(path, encoding="utf-8")
        else:
            atomic_write_ini(path, config)
        return config


def append_audit_log(log_path: Path, payload: Dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
