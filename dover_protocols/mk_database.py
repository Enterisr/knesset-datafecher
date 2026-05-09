from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .logger_config import get_logger


logger = get_logger(__name__)

_CACHE: dict[str, Dict[str, Dict[str, Any]]] = {}


def load_mks(mks_data_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load MK metadata from a `mks_data.json` file."""

    path = Path(mks_data_path)
    cache_key = str(path.resolve())
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("mks_data.json must contain a JSON object")
        _CACHE[cache_key] = data
        return data
    except FileNotFoundError:
        logger.error("MKs data file not found at %s", str(path))
        _CACHE[cache_key] = {}
        return {}
    except json.JSONDecodeError:
        logger.error("Error parsing MKs JSON data at %s", str(path))
        _CACHE[cache_key] = {}
        return {}


def get_mks(mks_data_path: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    """Get MK metadata.

    Defaults to `./mks_data.json` in the current working directory.
    """

    if mks_data_path is None:
        mks_data_path = Path.cwd() / "mks_data.json"
    return load_mks(mks_data_path)
