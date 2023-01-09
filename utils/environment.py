import os
from typing import List


def error(msg: str) -> None:
    raise RuntimeError(msg)

def has(key: str) -> bool:
    return os.getenv(key) is not None

def str(key: str, default: str = None, error_msg: str = None) -> str or None:
    val = os.getenv(key, default=default)
    if val is not None: return val
    if error_msg: error(error_msg.format(key))
    return None

def bool(key: str, default: bool = None, error_msg: str = None) -> bool or None:
    val = str(key, default=default, error_msg=error_msg)
    if val is None: return None
    return val.lower() == 'true'

def int(key: str, default: int = None, error_msg: str = None) -> int or None:
    val = str(key, default=default, error_msg=error_msg)
    if val is None: return None
    return int(val)

def enum(key: str, enum: List[str], case_sensitiv=False, default: str = None, error_msg: str = None) -> int or None:
    val_raw = str(key, default=default, error_msg=error_msg)
    if val_raw is None: return None
    val = val_raw

    if not case_sensitiv:
        enum = [it.lower() for it in enum]
        val = val.lower()

    if val in enum: return val_raw
    if error_msg: return error(error_msg)
    return None