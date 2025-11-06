import pandas as pd
import os
import json

def split_dataframe_by_keys(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ 自动根据后缀读取
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError("只支持 CSV 或 JSONL 文件")

    # 2️⃣ 定义输出路径
    paths = {
        "qas": os.path.join(output_dir, "qas.jsonl"),
        "qs": os.path.join(output_dir, "qs.jsonl"),
        "qa": os.path.join(output_dir, "qa.jsonl"),
        "other": os.path.join(output_dir, "other.jsonl"),
    }

    # 3️⃣ 打开写入流
    writers = {k: open(v, "w", encoding="utf-8") for k, v in paths.items()}

    # 4️⃣ 遍历每行，判断哪些字段不为空（分类逻辑只看 Q/A/S）
    for _, row in df.iterrows():
        q = row.get("question")
        a = row.get("answer")
        s = row.get("solution")

        has_q = pd.notna(q) and str(q).strip() != ""
        has_a = pd.notna(a) and str(a).strip() != ""
        has_s = pd.notna(s) and str(s).strip() != ""

        data = row.to_dict()  # ✅ 保留所有字段

        if has_q and has_a and has_s:
            writers["qas"].write(json.dumps(data, ensure_ascii=False) + "\n")
        elif has_q and has_s:
            writers["qs"].write(json.dumps(data, ensure_ascii=False) + "\n")
        elif has_q and has_a:
            writers["qa"].write(json.dumps(data, ensure_ascii=False) + "\n")
        else:
            writers["other"].write(json.dumps(data, ensure_ascii=False) + "\n")

    # 5️⃣ 关闭文件
    for f in writers.values():
        f.close()

    print("✅ 已保存到：", output_dir)


# 示例调用
if __name__ == "__main__":
    input_path = "/data1/hzh/vqa/math/real_analysis_4/vqa_filtered_qa_pairs.jsonl"
    output_dir = "/data1/hzh/vqa/sampled_jsonl/real_analysis_4"
    split_dataframe_by_keys(input_path, output_dir)
