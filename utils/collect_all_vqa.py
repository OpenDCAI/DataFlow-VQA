import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Any

#!/usr/bin/env python3
"""
collect_all_vqa.py

递归查找指定目录下（包括子目录）所有以 "_step_10.json" 结尾的文件，
将这些文件中的 item（支持多种常见结构）逐条写入一个 jsonl 文件，
并为每个 item 添加一个字段 "image_basedir" 表示该 json 文件所在的绝对文件夹路径。

Usage:
    python collect_all_vqa.py --root /path/to/search --out /path/to/out.jsonl
"""



def iter_items_from_file(path: Path) -> Iterable[Any]:
    """
    从一个 JSON 文件中提取 item。
    支持以下常见结构：
    - JSON array -> treat each element as an item
    - JSON dict with key 'items' (list) -> use that list
    - JSON dict (single item) -> yield the dict itself
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: cannot read/parse {path}: {e}", file=sys.stderr)
        return

    if isinstance(data, list):
        for it in data:
            yield it
    else:
        # unsupported root type
        print(f"Warning: unsupported JSON root type in {path}: {type(data)}", file=sys.stderr)
        return


def collect(root: Path, out_path: Path, suffix: str = "_step_10.json", overwrite: bool = False) -> int:
    if out_path.exists() and not overwrite:
        print(f"Error: output file {out_path} already exists. Use --overwrite to replace.", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for file_path in root.rglob(f"*{suffix}"):
            if not file_path.is_file():
                continue
            image_basedir = str(file_path.parent.resolve())
            for item in iter_items_from_file(file_path):
                if not isinstance(item, dict):
                    # wrap non-dict items into a dict under "value"
                    item = {"value": item}
                # add/overwrite image_basedir
                item["image_basedir"] = image_basedir
                item["subject"] = image_basedir.split("/")[-1]
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

    print(f"Collected {count} items into {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Collect items from *_step10.json files into a jsonl")
    parser.add_argument("--root", "-r", type=Path, default=Path("."), help="root folder to search")
    parser.add_argument("--out", "-o", type=Path, default=Path("collected.jsonl"), help="output jsonl file")
    parser.add_argument("--suffix", "-s", type=str, default="_step10.json", help="file name suffix to match")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output if exists")
    args = parser.parse_args()

    rc = collect(args.root, args.out, suffix=args.suffix, overwrite=args.overwrite)
    sys.exit(rc)


if __name__ == "__main__":
    main()