from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from .bad_dover_exception import BadDoverException
from .logger_config import get_logger
from .mk_database import get_mks


logger = get_logger(__name__)


def _token_sort_ratio_fallback(a: str, b: str) -> float:
    """Rough fallback if rapidfuzz isn't installed (0..100)."""
    import difflib

    a_sorted = " ".join(sorted(a.split()))
    b_sorted = " ".join(sorted(b.split()))
    return difflib.SequenceMatcher(None, a_sorted, b_sorted).ratio() * 100.0


try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore

    def _token_sort_ratio(a: str, b: str) -> float:
        return float(_rf_fuzz.token_sort_ratio(a, b))

except Exception:  # pragma: no cover

    def _token_sort_ratio(a: str, b: str) -> float:
        return _token_sort_ratio_fallback(a, b)


class DoverResolver:
    def __init__(
        self,
        min_ratio_for_match: float = 75.0,
        mks_data: Optional[Dict[str, Dict[str, Any]]] = None,
        mks_data_path: Optional[Union[str, Path]] = None,
    ):
        self.min_ratio = float(min_ratio_for_match)
        self.mks_by_id: Dict[str, Dict[str, Any]] = mks_data if mks_data is not None else get_mks(mks_data_path)
        self.mks_by_name: Dict[str, Dict[str, Any]] = self._build_name_index(self.mks_by_id)
        self.rapidfuzz_cache: dict[str, dict[str, Any]] = {}
        self.no_match_person: list[str] = []

    def _build_name_index(self, mks_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mks_by_name: Dict[str, Dict[str, Any]] = {}
        for mk_id, mk_data in mks_by_id.items():
            if isinstance(mk_data, dict) and "FirstName" in mk_data and "LastName" in mk_data:
                full_name = f"{mk_data.get('FirstName', '')} {mk_data.get('LastName', '')}".strip()
                if full_name:
                    mks_by_name[full_name] = {**mk_data, "mk_id": mk_id}
        return mks_by_name

    def remove_title_from_dover(self, dover_str: str) -> str:
        dover_str = dover_str.replace(' – מ"מ היו"ר', "")
        dover_str = dover_str.replace(' – היו"ר', "")
        dover_str = dover_str.replace('היו"ר ', "")
        dover_str = dover_str.replace('יושב-ראש הכנסת ', "")
        dover_str = dover_str.replace('יו"ר ', "")
        pattern = r"(שר|שרת)\s+\S+"
        dover_str = re.sub(pattern, "", dover_str)
        return dover_str

    def extract_name_key_from_dover(self, dover_str: str) -> str:
        dover_str = self.remove_title_from_dover(dover_str)
        match = re.match(r"^(.*?) \(", dover_str)
        name = match.group(1) if match else dover_str
        return name.strip()

    def fallback_to_fuzzy_match(self, name: str) -> Tuple[str, Dict[str, Any], float]:
        cached = self.rapidfuzz_cache.get(name)
        if cached is not None:
            return (cached["max_mk_key"], cached["max_sim_mk"], float(cached["max_ratio"]))

        max_ratio: float = 0.0
        max_sim_mk: Dict[str, Any] = {}
        max_mk_key: str = ""

        for mk_key, mk_meta in self.mks_by_name.items():
            ratio = _token_sort_ratio(name, mk_key)
            if ratio > max_ratio:
                max_ratio = ratio
                max_sim_mk = mk_meta
                max_mk_key = mk_key

        self.rapidfuzz_cache[name] = {
            "max_ratio": max_ratio,
            "max_sim_mk": max_sim_mk,
            "max_mk_key": max_mk_key,
        }

        if self.min_ratio > max_ratio:
            raise BadDoverException(f"Can't find mk to match {name}")

        return max_mk_key, max_sim_mk, max_ratio

    def resolve_mk(self, speaker: str, mks_in_meeting: Iterable[str]) -> tuple[str | None, Dict[str, Any] | None]:
        speaker_key = self.extract_name_key_from_dover(speaker)
        if speaker_key in set(mks_in_meeting):
            mk_meta = self.mks_by_name.get(speaker_key)
            if mk_meta is not None:
                return speaker_key, mk_meta

            try:
                rapidfuzz_match, mk_meta, ratio = self.fallback_to_fuzzy_match(speaker_key)
                logger.info("Fuzzy search for %s, found: %s (%.1f)", speaker_key, rapidfuzz_match, ratio)
                return rapidfuzz_match, mk_meta
            except BadDoverException:
                self.no_match_person.append(speaker_key)
                logger.error(
                    "Can't find match for %s (min ratio %.1f)",
                    speaker_key,
                    self.min_ratio,
                )

        return None, None
