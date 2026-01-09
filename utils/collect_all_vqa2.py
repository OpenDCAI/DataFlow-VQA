import argparse
import json
import sys
import re
import shutil
from pathlib import Path
from typing import Iterable, Any, Set

#!/usr/bin/env python3
"""
collect_all_vqa.py

递归查找指定目录下（包括子目录）所有以 "_step_10.json" 结尾的文件，
将这些文件中的 item（支持多种常见结构）逐条写入一个 jsonl 文件，
并为每个 item 添加一个字段 "image_basedir" 表示该 json 文件所在的绝对文件夹路径。

Usage:
    python collect_all_vqa.py --root /path/to/search --out /path/to/out.jsonl
"""



def extract_image_refs(text: str) -> Set[str]:
    """
    从文本中提取图片引用路径，格式为 ![alt_text](question_images/filename.jpg) 或 ![alt_text](answer_images/filename.jpg)
    返回所有找到的图片路径集合
    """
    if not isinstance(text, str):
        return set()

    pattern = r'!\[[^\]]*\]\((question_images|answer_images)/([^)]+)\)'
    matches = re.findall(pattern, text)
    return {f"{prefix}/{filename}" for prefix, filename in matches}


def copy_images_to_folder(image_refs: Set[str], image_basedir: Path, target_images_dir: Path, subject: str) -> dict:
    """
    将图片从源目录复制到目标目录，保持 question_images 和 answer_images 的文件夹结构
    返回一个映射字典：原路径 -> 新路径
    """
    path_mapping = {}

    for image_ref in image_refs:
        # image_ref 格式: question_images/filename.jpg 或 answer_images/filename.jpg
        source_path = image_basedir / image_ref

        if source_path.exists() and source_path.is_file():
            # 目标路径：保持文件夹结构 images/{subject}/question_images/ 或 images/{subject}/answer_images/
            # 提取文件夹名（question_images 或 answer_images）
            folder_name = image_ref.split('/')[0]  # question_images 或 answer_images
            filename = source_path.name

            target_subject_dir = target_images_dir / subject
            target_folder_dir = target_subject_dir / folder_name
            target_folder_dir.mkdir(parents=True, exist_ok=True)

            target_path = target_folder_dir / filename

            # 如果文件已存在且相同，跳过
            if target_path.exists():
                if target_path.stat().st_size == source_path.stat().st_size:
                    path_mapping[image_ref] = f"images/{subject}/{image_ref}"
                    continue

            try:
                shutil.copy2(source_path, target_path)
                path_mapping[image_ref] = f"images/{subject}/{image_ref}"
            except Exception as e:
                print(f"Warning: failed to copy {source_path} to {target_path}: {e}", file=sys.stderr)
        else:
            print(f"Warning: image not found: {source_path}", file=sys.stderr)

    return path_mapping


def update_image_refs(text: str, path_mapping: dict) -> str:
    """
    更新文本中的图片引用路径
    """
    if not isinstance(text, str):
        return text

    def replace_ref(match):
        alt_text = match.group("alt")
        prefix = match.group("prefix")
        filename = match.group("filename")
        old_ref = f"{prefix}/{filename}"
        if old_ref in path_mapping:
            new_path = path_mapping[old_ref]
            return f"![{alt_text}]({new_path})"
        return match.group(0)

    pattern = r'!\[(?P<alt>[^\]]*)\]\((?P<prefix>question_images|answer_images)/(?P<filename>[^)]+)\)'
    return re.sub(pattern, replace_ref, text)


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

    # 创建图片文件夹（在输出文件同目录下）
    target_images_dir = out_path.parent / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    # 用于跟踪每个目录的图片引用和路径映射
    dir_image_refs = {}  # image_basedir -> set of image_refs
    dir_path_mapping = {}  # image_basedir -> path_mapping dict
    items_data = []  # 临时存储所有item数据

    # 第一遍：收集所有item和图片引用
    for file_path in root.rglob(f"*{suffix}"):
        if not file_path.is_file():
            continue
        image_basedir = Path(file_path.parent.resolve())

        if image_basedir not in dir_image_refs:
            dir_image_refs[image_basedir] = set()

        for item in iter_items_from_file(file_path):
            if not isinstance(item, dict):
                item = {"value": item}

            # 提取图片引用
            question = item.get("question", "")
            answer = item.get("answer", "")
            solution = item.get("solution", "")

            image_refs = set()
            image_refs.update(extract_image_refs(question))
            image_refs.update(extract_image_refs(answer))
            image_refs.update(extract_image_refs(solution))

            dir_image_refs[image_basedir].update(image_refs)

            question_has_image = len(extract_image_refs(question)) > 0

            # 保存item数据用于后续处理
            items_data.append({
                "item": item,
                "image_basedir": image_basedir,
                "new_image_basedir": Path(out_path.parent.resolve()),
                "subject": image_basedir.name,
                "image_refs": image_refs,
                "question_has_image": question_has_image
            })

    # 复制所有图片到目标文件夹
    total_refs = sum(len(refs) for refs in dir_image_refs.values())
    print(f"Copying images from {len(dir_image_refs)} directories ({total_refs} total references)...")

    for image_basedir, image_refs in dir_image_refs.items():
        if image_refs:
            subject = image_basedir.name  # 获取文件夹名，如 abstract_algebra_1
            path_mapping = copy_images_to_folder(image_refs, image_basedir, target_images_dir, subject)
            dir_path_mapping[image_basedir] = path_mapping

    # 第二遍：写入输出文件并更新图片引用
    count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for data in items_data:
            item = data["item"].copy()
            image_basedir = data["image_basedir"]

            # 更新图片引用
            path_mapping = dir_path_mapping.get(image_basedir, {})

            if "question" in item:
                item["question"] = update_image_refs(item["question"], path_mapping)
            if "answer" in item:
                item["answer"] = update_image_refs(item["answer"], path_mapping)
            if "solution" in item:
                item["solution"] = update_image_refs(item["solution"], path_mapping)

            # 添加元数据
            item["image_basedir"] = str(data["new_image_basedir"])
            item["subject"] = data["subject"]
            item["question_has_image"] = data["question_has_image"]

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    print(f"Collected {count} items into {out_path}")
    print(f"Images copied to {target_images_dir}")
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