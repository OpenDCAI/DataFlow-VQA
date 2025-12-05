import os
import json
import re
from collections import defaultdict

def process_subject_folder(root_dir, subject_prefix):
    """
    遍历指定根目录，找到以学科前缀开头的子文件夹，
    并对其中的特定文件进行统计。

    Args:
        root_dir (str): 包含所有子文件夹（如 math_1, physics_1）的父目录。
        subject_prefix (str): 学科名称前缀（如 'math'）。

    Returns:
        tuple: (
            dict: 存储每个子文件夹统计结果的字典,
            dict: 存储所有子文件夹总计结果的字典
        )
    """

    # 1. 目标文件后缀
    TARGET_SUFFIXES = [
        '_step1.json',
        '_step2.json',
        '_step5.json',
        '_step10.json'
    ]

    # 2. 正则表达式模式 (用于 step2 和 step10 文件)
    # 查找 "question" 字段中包含 "![" 和 "]" 的模式
    # 即 "question":".*!\[.*\].*"
    REGEX_PATTERN = re.compile(r".*!\[.*?\].*") # ? 表示非贪婪匹配

    # 初始化存储结构
    folder_results = defaultdict(lambda: {
        'len_step1': 0,
        'len_step2': 0,
        'len_step5': 0,
        'len_step10': 0,
        'regex_count_step2': 0,
        'regex_count_step10': 0
    })

    # 遍历 root_dir 下的所有项
    for item_name in os.listdir(root_dir):
        # 检查是否为目录且以目标学科开头 (e.g., 'math_1')
        current_path = os.path.join(root_dir, item_name)
        if os.path.isdir(current_path) and item_name.startswith(subject_prefix):
            # item_name 就是子文件夹名 (e.g., 'math_1')
            sub_folder_name = item_name

            # 遍历子文件夹中的文件
            for file_name in os.listdir(current_path):

                # 检查文件是否是我们需要的目标文件
                for suffix in TARGET_SUFFIXES:
                    if file_name.endswith(suffix):
                        file_path = os.path.join(current_path, file_name)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            # 目标文件都是一个 list
                            if isinstance(data, list):
                                list_length = len(data)

                                # 统计 1. 列表长度
                                if suffix == '_step1.json':
                                    folder_results[sub_folder_name]['len_step1'] = list_length
                                elif suffix == '_step2.json':
                                    folder_results[sub_folder_name]['len_step2'] = list_length
                                    # 统计 2. 正则表达式数量 (仅 step2)
                                    count_step2 = sum(
                                        1 for item in data
                                        if isinstance(item, dict) and 'question' in item
                                        and REGEX_PATTERN.match(item['question'])
                                    )
                                    folder_results[sub_folder_name]['regex_count_step2'] = count_step2
                                elif suffix == '_step5.json':
                                    folder_results[sub_folder_name]['len_step5'] = list_length
                                elif suffix == '_step10.json':
                                    folder_results[sub_folder_name]['len_step10'] = list_length
                                    # 统计 2. 正则表达式数量 (仅 step10)
                                    count_step10 = sum(
                                        1 for item in data
                                        if isinstance(item, dict) and 'question' in item
                                        and REGEX_PATTERN.match(item['question'])
                                    )
                                    folder_results[sub_folder_name]['regex_count_step10'] = count_step10

                        except (json.JSONDecodeError, IOError) as e:
                            print(f"Error processing file {file_path}: {e}")
                        break # 跳出 suffix 循环，处理下一个文件

    # 计算总计
    total_results = {
        'len_step1': sum(r['len_step1'] for r in folder_results.values()),
        'len_step2': sum(r['len_step2'] for r in folder_results.values()),
        'len_step5': sum(r['len_step5'] for r in folder_results.values()),
        'len_step10': sum(r['len_step10'] for r in folder_results.values()),
        'regex_count_step2': sum(r['regex_count_step2'] for r in folder_results.values()),
        'regex_count_step10': sum(r['regex_count_step10'] for r in folder_results.values()),
    }

    return folder_results, total_results

def display_results(results, total_results):
    """
    格式化并输出统计结果。
    """
    # 定义输出表格的列名
    headers = [
        "Folder",
        "L_S1",
        "L_S2",
        "L_S5",
        "L_S10",
        "R_S2",
        "R_S10"
    ]
    # 格式化字符串，用于对齐输出
    # 假设文件夹名最长10位，数字最多5位
    row_format = "{:<10} | {:>4} | {:>4} | {:>4} | {:>4} | {:>4} | {:>5}"

    print("\n--- 📊 详细统计结果 (L: List Length, R: Regex Count) ---")
    print(row_format.format(*headers))
    print("-" * 48)

    # 1. 输出每个子文件夹的结果
    for folder, res in sorted(results.items()):
        output_data = [
            folder,
            res['len_step1'],
            res['len_step2'],
            res['len_step5'],
            res['len_step10'],
            res['regex_count_step2'],
            res['regex_count_step10']
        ]
        print(row_format.format(*output_data))

    # 2. 输出总计结果
    print("-" * 48)
    total_data = [
        "TOTAL",
        total_results['len_step1'],
        total_results['len_step2'],
        total_results['len_step5'],
        total_results['len_step10'],
        total_results['regex_count_step2'],
        total_results['regex_count_step10']
    ]
    print(row_format.format(*total_data))
    print("----------------------------------------------------------")


# --- 📌 使用示例 ---
if __name__ == '__main__':
    # **请根据您的实际情况修改以下两个变量**
    # 您的根文件夹路径，即包含 math_1, math_2, physics_1... 的目录
    ROOT_DIRECTORY = '/data0/djw/VQA_ready_data'
    # 您要统计的学科前缀 (例如 'math')
    SUBJECT = 'thermodynamics_and_statistical_physics'

    # 检查根目录是否存在
    if not os.path.isdir(ROOT_DIRECTORY):
        print(f"ERROR: Root directory not found at '{ROOT_DIRECTORY}'. Please update the ROOT_DIRECTORY variable.")
    else:
        print(f"Starting analysis for subject '{SUBJECT}' in directory '{ROOT_DIRECTORY}'...")
        individual_results, aggregated_totals = process_subject_folder(
            root_dir=ROOT_DIRECTORY,
            subject_prefix=SUBJECT
        )

        if individual_results:
            display_results(individual_results, aggregated_totals)
        else:
            print(f"No folders starting with '{SUBJECT}' found in the root directory.")