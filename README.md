# dover_protocols

Small, self-contained Python module extracted from this repo for:

1. Downloading Knesset committee protocol documents (DOC) via the Knesset OData API
2. Converting them to UTF-8 text (via LibreOffice)
3. Saving per-protocol JSON files
4. Extracting utterances from the text using Dover tags and resolving speakers to MK metadata

## Requirements

- Python 3.10+
- `requests`
- `requests-cache`
- Optional: `rapidfuzz` (better name matching; falls back to stdlib if missing)
- LibreOffice installed and `soffice.com` available on PATH (Windows).

## Quick usage

Download committee protocols to partitioned JSON files:

```python
from dover_protocols import KnessetDataFetcher

fetcher = KnessetDataFetcher(knesset_num=25, force_refresh=False)
fetcher.process_knesset_data()  # writes `mks_data.json` and `committee_data/part_*/<doc>.json`

# Download only a specific committee (e.g. Vaadat Ksafim / Finance Committee):
fetcher = KnessetDataFetcher(knesset_num=25, committee_filter="vaadat ksafim", force_refresh=False)
fetcher.process_knesset_data()
```

Extract utterances into a parallel folder structure:

```python
from dover_protocols import process_protocols

process_protocols(
    output_folder="committee_data",
    utterances_folder="utterances",
    force_refresh=False,
)
```

## Notes

- `DoverResolver` defaults to loading `mks_data.json` from the current working directory.
  If you store it elsewhere, pass `mks_data_path=` when constructing `DoverResolver`.
