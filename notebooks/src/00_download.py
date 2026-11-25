"""Download the Sen1Floods11 v1.1 hand-labelled Sentinel-2 chips, labels and JRC layer.

Source: public Google Cloud Storage bucket `sen1floods11` (no account needed).
Each file is checked against the bucket's MD5, then hashed with SHA-256. The bucket
listing is saved before and after the download so that any change during the
download is visible. Files are written once and set read-only.

Run:  python notebooks/src/00_download.py
"""
import base64
import csv
import hashlib
import json
import os
import pathlib
import stat
import time
from datetime import datetime, timezone

import requests

PROJECT = pathlib.Path(os.environ.get("PAPER_ROOT", pathlib.Path(__file__).resolve().parents[2]))
RAW = PROJECT / "data" / "raw" / "sen1floods11_v1.1"
RECORDS = PROJECT / "data" / "records"
BUCKET = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
LAYERS = ("S2Hand", "LabelHand", "JRCWaterHand")
PREFIX = "v1.1/data/flood_events/HandLabeled/"
EXTRA = ("v1.1/Sen1Floods11_Metadata.geojson",
         "v1.1/splits/flood_handlabeled/flood_train_data.csv",
         "v1.1/splits/flood_handlabeled/flood_valid_data.csv",
         "v1.1/splits/flood_handlabeled/flood_test_data.csv",
         "v1.1/splits/flood_handlabeled/flood_bolivia_data.csv")


def list_objects(prefix):
    """Return every object under a bucket prefix, following page tokens."""
    items, token = [], None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        js = requests.get(BUCKET, params=params, timeout=60).json()
        items += js.get("items", [])
        token = js.get("nextPageToken")
        if not token:
            return items


def wanted():
    """Objects to download: the three hand-labelled layers plus metadata and split files."""
    objs = []
    for layer in LAYERS:
        objs += list_objects(PREFIX + layer + "/")
    for name in EXTRA:
        objs += [o for o in list_objects(name) if o["name"] == name]
    return sorted(objs, key=lambda o: o["name"])


def fetch(obj, target):
    """Download one object, verify its MD5 against the bucket record, return SHA-256."""
    expected_md5 = base64.b64decode(obj["md5Hash"]).hex()
    for attempt in range(5):
        try:
            r = requests.get(obj["mediaLink"], timeout=120)
            r.raise_for_status()
            data = r.content
            if hashlib.md5(data).hexdigest() == expected_md5:
                break
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"download or MD5 check failed: {obj['name']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    os.chmod(target, stat.S_IREAD)
    return hashlib.sha256(data).hexdigest(), expected_md5


def listing_record(objs):
    return [{"name": o["name"], "size": int(o["size"]), "md5": o["md5Hash"], "generation": o["generation"],
             "updated": o["updated"]} for o in objs]


def main():
    RECORDS.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    before = wanted()
    (RECORDS / "bucket_listing_before.json").write_text(json.dumps(listing_record(before), indent=1), encoding="utf-8")
    rows = []
    for i, obj in enumerate(before, 1):
        target = RAW / obj["name"].removeprefix("v1.1/")
        if target.exists():
            sha = hashlib.sha256(target.read_bytes()).hexdigest()
            md5 = hashlib.md5(target.read_bytes()).hexdigest()
            if md5 != base64.b64decode(obj["md5Hash"]).hex():
                raise RuntimeError(f"existing file differs from bucket: {target}")
        else:
            sha, md5 = fetch(obj, target)
        rows.append({"path": str(target.relative_to(PROJECT)).replace("\\", "/"), "bytes": int(obj["size"]),
                     "md5": md5, "sha256": sha, "generation": obj["generation"]})
        if i % 100 == 0:
            print(f"{i}/{len(before)}", flush=True)
    after = wanted()
    (RECORDS / "bucket_listing_after.json").write_text(json.dumps(listing_record(after), indent=1), encoding="utf-8")
    changed = {o["name"] for o in before} ^ {o["name"] for o in after}
    changed |= {a["name"] for a, b in zip(before, after) if a["generation"] != b["generation"]}
    with open(RECORDS / "file_hashes.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "files": len(rows), "bytes": sum(r["bytes"] for r in rows),
               "bucket_changed_during_download": sorted(changed)}
    (RECORDS / "download_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
