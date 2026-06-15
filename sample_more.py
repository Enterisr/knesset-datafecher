import glob, json, sys, random, csv, os
sys.stdout.reconfigure(encoding='utf-8')

all_files = sorted(glob.glob(os.path.join("committee_data", "**", "*.json"), recursive=True))
print(f"Total files: {len(all_files)}")

with open("committee_subject_labeling_signal.csv", encoding="utf-8-sig") as f:
    done = set(r["doc_id"] for r in csv.DictReader(f))
print(f"Already done: {len(done)}")

remaining = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] not in done]
print(f"Remaining: {len(remaining)}")

random.seed(99)
sample = random.sample(remaining, min(70, len(remaining)))
for f in sample:
    with open(f, encoding="utf-8") as fh:
        d = json.load(fh)
    print(f"{d['doc_id']}|{len(d['utterances'])}|{d['date'][:10]}|{d['title'][:70]}")
