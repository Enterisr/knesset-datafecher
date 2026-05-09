from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from subprocess import run
from urllib.parse import urlparse

import requests
import requests_cache

from .duplicate_checker import check_for_duplicate_files
from .logger_config import get_logger
from .partition_handler import PartitionHandler


class KnessetDataFetcher:
    """Fetch and process Knesset committee protocols into JSON files."""

    CACHE_FILE = "knesset_cache.sqlite"
    TEMP_RESOURCE_FOLDER = "temp"
    OUTPUT_FOLDER = "committee_data"
    FILE_BUFFER_SIZE = 8192
    COMMITTEE_SESSION_STR = "KNS_DocumentCommitteeSession"
    COMMITTEES_DATA_URI = "https://knesset.gov.il/OdataV4/ParliamentInfo/KNS_Committee"
    COMMITTEES_PAGE_SIZE = 200
    MAX_CAST_TRIES_FOR_DOC = 10

    COMMITTEE_NAME_ALIASES: dict[str, str] = {
        # Common transliterations / English names -> official Hebrew committee name
        "vaadat ksafim": "ועדת הכספים",
        "vaadat hakasafim": "ועדת הכספים",
        "finance committee": "ועדת הכספים",
    }

    def __init__(
        self,
        knesset_num: int,
        force_refresh: bool = False,
        to_save_txt: bool = False,
        output_folder: str | None = None,
        temp_folder: str | None = None,
        committee_filter: str | None = None,
    ):
        self.knesset_num = knesset_num
        self.force_refresh = force_refresh
        self.to_save_txt = to_save_txt
        self.committee_filter = committee_filter

        if output_folder is not None:
            self.OUTPUT_FOLDER = output_folder
        if temp_folder is not None:
            self.TEMP_RESOURCE_FOLDER = temp_folder

        self.committees: dict[int, dict] = {}
        self.mks: dict[int, dict] = {}
        self._missing_committee_ids: set[int] = set()

        self.partition_handler = PartitionHandler()
        self.logger = get_logger(__name__)

        requests_cache.install_cache(self.CACHE_FILE, backend="sqlite", expire_after=3600)

        self._init_folders()
        self._load_committees_data()

    @staticmethod
    def _normalize_committee_name(value: str) -> str:
        value = (value or "").strip()
        value = re.sub(r"[_\-–—]+", " ", value)
        value = re.sub(r"\s+", " ", value)
        return value.lower()

    def _committee_allowed(self, committee_name: str) -> bool:
        """Return True if this committee should be processed under the current filter."""

        if not self.committee_filter:
            return True

        raw_filter = self._normalize_committee_name(self.committee_filter)
        raw_filter = self.COMMITTEE_NAME_ALIASES.get(raw_filter, raw_filter)

        normalized_committee = self._normalize_committee_name(committee_name)
        normalized_filter = self._normalize_committee_name(raw_filter)

        # Substring match keeps it flexible (e.g. pass "כספים" or "ועדת הכספים")
        return normalized_filter in normalized_committee

    def _init_folders(self) -> None:
        os.makedirs(self.TEMP_RESOURCE_FOLDER, exist_ok=True)
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)

    def _load_committees_data(self) -> None:
        skip = 0
        while True:
            uri = f"{self.COMMITTEES_DATA_URI}?$top={self.COMMITTEES_PAGE_SIZE}&$skip={skip}"
            response = requests.get(uri)
            response.raise_for_status()
            committees_list = response.json().get("value", [])
            if not committees_list:
                break

            for committee in committees_list:
                committee_id = committee.get("Id")
                if committee_id is not None:
                    self.committees[int(committee_id)] = committee

            skip += self.COMMITTEES_PAGE_SIZE

    def _fetch_committee_by_id(self, committee_id: int) -> dict | None:
        uri = f"{self.COMMITTEES_DATA_URI}?$filter=Id%20eq%20{committee_id}&$top=1"
        try:
            response = requests.get(uri)
            response.raise_for_status()
            value = response.json().get("value", [])
            if value:
                return value[0]
        except Exception as e:
            self.logger.warning(
                "Failed fetching committee metadata for CommitteeID=%s: %s",
                committee_id,
                str(e),
            )
        return None

    def _get_committee_name(self, committee_id: int | None) -> str:
        if committee_id is None:
            return "unknown_committee"

        committee = self.committees.get(int(committee_id))
        if committee is None:
            committee = self._fetch_committee_by_id(int(committee_id))
            if committee is not None:
                self.committees[int(committee_id)] = committee
            else:
                if int(committee_id) not in self._missing_committee_ids:
                    self.logger.warning(
                        "Unknown CommitteeID=%s encountered; using placeholder name",
                        committee_id,
                    )
                    self._missing_committee_ids.add(int(committee_id))
                self.committees[int(committee_id)] = {
                    "Id": int(committee_id),
                    "Name": f"unknown_committee_{committee_id}",
                }

        return (
            str(self.committees[int(committee_id)].get("Name", "unknown_committee")).strip()
            or "unknown_committee"
        )

    def read_doc_as_txt(self, doc: str) -> str:
        """Convert a document to UTF-8 text using LibreOffice (`soffice.com`)."""

        cmd = [
            "soffice.com",
            "--convert-to",
            "txt:Text (encoded):UTF8",
            "--outdir",
            self.TEMP_RESOURCE_FOLDER,
            doc,
        ]
        run(cmd, check=True)

        doc_path = Path(doc)
        txt_doc_path = doc_path.stem + ".txt"
        output_file = Path(os.path.join(self.TEMP_RESOURCE_FOLDER, txt_doc_path))

        with open(output_file, "r", encoding="utf-8") as f:
            text_content = f.read()

        if not self.to_save_txt:
            output_file.unlink(missing_ok=True)

        return text_content

    def extract_json_path(self, meta: dict) -> str:
        doc_id = Path(urlparse(meta["FilePath"]).path).stem
        folder = os.path.join(self.OUTPUT_FOLDER, self.partition_handler.get_folder())
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{doc_id}.json")

    def save_doc_as_json(self, text: str, meta: dict, out_path: str) -> None:
        doc_id = Path(urlparse(meta["FilePath"]).path).stem
        committee_name = meta.get("CommitteeName", "unknown_committee").strip().replace(" ", "_")
        date = meta.get("SessionDate", None)

        data = {
            "knesset_num": self.knesset_num,
            "committee": committee_name,
            "doc_id": doc_id,
            "date": date,
            "source_file": meta["FilePath"],
            "text": text,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def read_resource_from_remote(self, uri: str) -> str:
        cleaned_url = re.sub(r"(?<!:)//", "/", uri)
        with requests_cache.disabled():
            response = requests.get(cleaned_url)
            response.raise_for_status()

        file_name = os.path.basename(urlparse(cleaned_url).path)
        current_path = os.path.join(self.TEMP_RESOURCE_FOLDER, file_name)
        with open(current_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=self.FILE_BUFFER_SIZE):
                if chunk:
                    f.write(chunk)
        return os.path.abspath(current_path)

    def remove_resource_after_reading(self, doc_path: str) -> bool:
        if os.path.exists(doc_path):
            try:
                os.remove(doc_path)
                return True
            except OSError as e:
                self.logger.error("Error removing file %s: %s", doc_path, str(e))
        return False

    def process_document(self, doc: dict, committee_name: str, date: str, tries: int = 0) -> None:
        doc["CommitteeName"] = committee_name
        doc["SessionDate"] = date
        doc_path = ""
        try:
            out_path = self.extract_json_path(doc)
            if self.force_refresh or not os.path.exists(out_path):
                doc_path = self.read_resource_from_remote(doc["FilePath"])
                text = self.read_doc_as_txt(doc_path)
                self.save_doc_as_json(text, doc, out_path)
        except Exception as e:
            if tries < self.MAX_CAST_TRIES_FOR_DOC:
                self.logger.info("Error processing %s, retrying (%s)", doc.get("FilePath"), str(e))
                self.process_document(doc, committee_name, date, tries=tries + 1)
            else:
                self.logger.error("Error processing %s OUT OF TRIES", doc.get("FilePath"))
        finally:
            if doc_path:
                self.remove_resource_after_reading(doc_path)

    def fetch_mks_data(self) -> dict:
        uri_for_person_id = (
            "https://knesset.gov.il/OdataV4/ParliamentInfo/KNS_PersonToPosition?"
            f"$filter=KnessetNum%20eq%20{self.knesset_num}%20and%20FactionName%20ne%20null&$expand=KNS_Person"
        )
        res = requests.get(uri_for_person_id)
        res.raise_for_status()
        mks_list = res.json().get("value", [])
        for mk in mks_list:
            self.mks[int(mk["PersonID"])] = {
                "Id": mk["KNS_Person"]["Id"],
                "FirstName": mk["KNS_Person"].get("FirstName"),
                "Email": mk["KNS_Person"].get("Email"),
                "LastName": mk["KNS_Person"].get("LastName"),
                "FactionName": mk.get("FactionName"),
                "FactionID": mk.get("FactionID"),
            }

        self.save_mks_to_file(self.mks)
        return self.mks

    def save_mks_to_file(self, mks_data: dict, file_path: str = "mks_data.json") -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(mks_data, f, ensure_ascii=False, indent=2)
        self.logger.info("MKs data saved to %s", file_path)

    def fetch_all_committees_from_knesset(self) -> None:
        debug = os.getenv("DEBUG", "false").lower() == "true"

        page_size = 50
        skip = 0

        while True:
            self.logger.debug(
                "Fetching committee sessions from Knesset %s: skip=%s, limit=%s",
                self.knesset_num,
                skip,
                page_size,
            )

            is_end = self.fetch_paginated_committees_from_knesset(page_size, skip)
            if not is_end or debug:
                break
            skip += page_size

        if debug:
            self.logger.info("Debug mode: Only fetched first page")

    def build_committees_uri(self, top: int, skip: int) -> str:
        expand_part = "$expand=KNS_CmtSessionItem%2CKNS_DocumentCommitteeSession"
        filter_part = f"$filter=KnessetNum%20eq%20{self.knesset_num}"
        pagination_part = f"$top={top}&$skip={skip}&$orderby=ID"
        return (
            "https://knesset.gov.il/OdataV4/ParliamentInfo/KNS_CommitteeSession?"
            f"{filter_part}&{expand_part}&{pagination_part}"
        )

    def fetch_paginated_committees_from_knesset(self, top: int, skip: int) -> bool:
        uri = self.build_committees_uri(top, skip)
        response = requests.get(uri)
        response.raise_for_status()
        committees_data = response.json().get("value", [])
        if len(committees_data) == 0:
            return False

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for session in committees_data:
                committee_name = self._get_committee_name(session.get("CommitteeID"))
                date = session.get("StartDate", None)

                if not self._committee_allowed(committee_name):
                    continue

                if self.COMMITTEE_SESSION_STR in session:
                    for doc in session[self.COMMITTEE_SESSION_STR]:
                        if doc.get("ApplicationDesc") == "DOC" and doc.get("GroupTypeID") == 23:
                            futures.append(
                                executor.submit(self.process_document, doc, committee_name, date)
                            )

            for future in as_completed(futures):
                future.result()

        return True

    def process_knesset_data(self) -> None:
        self.fetch_mks_data()
        self.fetch_all_committees_from_knesset()
        self.logger.info("Checking for duplicate JSON files...")
        check_for_duplicate_files(self.OUTPUT_FOLDER)
