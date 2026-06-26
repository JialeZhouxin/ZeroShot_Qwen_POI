#!/usr/bin/env python3
"""从 data.zip 解压数据到 ./data/ 目录，或从 processed_sthgcn data 生成。

用法：
  python prepare_data.py                    # 解压当前目录下的 data.zip
  python prepare_data.py --source <path>   # 从 processed_sthgcn data 目录生成
"""

import argparse
import csv
import os
import shutil
import sys
import zipfile
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

EXPECTED_COLS = [
    "UserId", "PoiId", "PoiCategoryName",
    "Latitude", "Longitude", "UTCTimeOffset",
    "pseudo_session_trajectory_id", "SplitTag",
]


def extract_zip(zip_path):
    """解压 data.zip 到 ./data/"""
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found. Download it from the link in README.md.")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

    # zip 内结构: data/CA/CA_train.csv ...
    # 如果提取后是 ./data/data/CA/... 则展平一级
    nested = os.path.join(DATA_DIR, "data")
    if os.path.isdir(nested):
        for name in os.listdir(nested):
            shutil.move(os.path.join(nested, name), os.path.join(DATA_DIR, name))
        os.rmdir(nested)

    print(f"Extracted {zip_path} -> {DATA_DIR}/")


def convert_from_sthgcn(source_dir):
    """从 processed_sthgcn data 目录生成 train/val CSV。

    将 sample.csv 按 SplitTag 拆分为 {D}_train.csv / {D}_val.csv，
    列名映射为 load_dataset 所需格式。
    """
    COLUMN_MAP = {
        "UTCTimeOffset": "local_time",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }

    for dataset in ["NYC", "TKY", "CA"]:
        src = os.path.join(source_dir, dataset, "sample.csv")
        if not os.path.exists(src):
            print(f"WARNING: {src} not found, skipping {dataset}")
            continue

        out_dir = os.path.join(DATA_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)

        with open(src, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Rename columns
        renamed_rows = []
        for r in rows:
            nr = {}
            for k, v in r.items():
                new_k = COLUMN_MAP.get(k, k)
                nr[new_k] = v
            renamed_rows.append(nr)

        all_cols = list(renamed_rows[0].keys())

        # Write train
        train_rows = [r for r in renamed_rows if r.get("SplitTag", "").strip() == "train"]
        with open(os.path.join(out_dir, f"{dataset}_train.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=all_cols)
            w.writeheader()
            w.writerows(train_rows)

        # Write val (validation + test splits)
        val_rows = [r for r in renamed_rows if r.get("SplitTag", "").strip() in ("validation", "test")]
        with open(os.path.join(out_dir, f"{dataset}_val.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=all_cols)
            w.writeheader()
            w.writerows(val_rows)

        print(f"{dataset}: {len(train_rows)} train + {len(val_rows)} val -> {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for ZeroShot_Qwen")
    parser.add_argument(
        "--source", default=None,
        help="Path to processed_sthgcn data root (e.g. 'data/processed_sthgcn data'). "
             "If omitted, expects data.zip in script directory and extracts it."
    )
    parser.add_argument(
        "--zip", default="data.zip",
        help="Path to data.zip (default: ./data.zip)"
    )
    args = parser.parse_args()

    if args.source:
        convert_from_sthgcn(args.source)
    else:
        zip_path = os.path.join(SCRIPT_DIR, args.zip)
        extract_zip(zip_path)

    # Verify
    for ds in ["NYC", "TKY", "CA"]:
        train = os.path.join(DATA_DIR, ds, f"{ds}_train.csv")
        val = os.path.join(DATA_DIR, ds, f"{ds}_val.csv")
        if not os.path.exists(train) or not os.path.exists(val):
            print(f"WARNING: {ds} data incomplete. Check data source.")
        else:
            print(f"  {ds}: train={os.path.getsize(train)//1024}KB, val={os.path.getsize(val)//1024}KB")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
