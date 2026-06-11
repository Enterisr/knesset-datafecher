from dover_protocols import KnessetDataFetcher

def main() -> None:
        
    fetcher = KnessetDataFetcher(knesset_num=25, committee_filter="vaadat ksafim", force_refresh=True)
    fetcher.process_knesset_data()

if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()