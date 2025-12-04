import json
import matplotlib
import numpy as np
import tiktoken
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

input_file = "/data1/djw/DataFlow-VQA/rollout_cache/all_math_vqa_only/math-Qwen3-8B-Instruct_step5.json"
input_solution_file = "/data1/djw/DataFlow-VQA/rollout_cache/all_math_vqa_only/math-Qwen3-8B-Instruct_step4.json"

# 根据每个item的accuracy项画分布图
def draw_acc_distribution(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    accuracies = [item['accuracy'] for item in data if 'accuracy' in item]

    # 设置matplotlib后端为agg，避免在没有显示设备的环境中出错
    matplotlib.use('agg')

    plt.figure(figsize=(10, 6))
    if len(accuracies) == 0:
        plt.text(0.5, 0.5, 'No accuracy data', ha='center')
    else:
        weights = np.ones(len(accuracies)) / len(accuracies)
        plt.hist(accuracies, bins=30, weights=weights, edgecolor='black', alpha=0.7)
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    plt.title(f'Distribution of Accuracies: {file_name.split("_")[0]}')
    plt.xlabel('Accuracy')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.75)
    
    output_file = input_file.replace('.json', '_distribution.png')
    plt.savefig(output_file)
    print(f"Distribution plot saved to {output_file}")
    
def draw_solution_length_distribution_by_result(input_file, key):
    with open(input_file, 'r') as f:
        data = json.load(f)

    tokenizer = tiktoken.get_encoding("o200k_base")
    
    lengths_correct = []
    lengths_incorrect = []
    
    for item in tqdm(data, desc="Processing items", unit="it"):
        if key in item and 'answer_match_result' in item:
            tokenized = tokenizer.encode(item[key])
            length = len(tokenized)
            
            if item['answer_match_result']:
                lengths_correct.append(length)
            else:
                lengths_incorrect.append(length)

    # 设置后端
    matplotlib.use('agg')

    plt.figure(figsize=(10, 6))
    
    if len(lengths_correct) == 0 and len(lengths_incorrect) == 0:
        plt.text(0.5, 0.5, 'No solution length data', ha='center')
    else:
        # 1. 计算平均值
        mean_correct = np.mean(lengths_correct) if lengths_correct else 0
        mean_incorrect = np.mean(lengths_incorrect) if lengths_incorrect else 0

        # 2. 准备标签，包含平均值信息
        label_correct = f'Correct (Avg: {mean_correct:.1f})'
        label_incorrect = f'Incorrect (Avg: {mean_incorrect:.1f})'

        data_to_plot = [lengths_correct, lengths_incorrect]
        colors = ['#2ca02c', '#d62728'] # 绿，红
        labels = [label_correct, label_incorrect]

        # 绘制堆叠直方图
        plt.hist(data_to_plot, 
                 bins=30, 
                 stacked=False, 
                 color=colors, 
                 label=labels, 
                 edgecolor='black', 
                 alpha=0.7,
                 log=True)
        
        # 3. 添加垂直虚线标记平均值位置 (可选，增强可视性)
        if lengths_correct:
            plt.axvline(mean_correct, color='darkgreen', linestyle='dashed', linewidth=1.5)
        if lengths_incorrect:
            plt.axvline(mean_incorrect, color='darkred', linestyle='dashed', linewidth=1.5)

    plt.title(f'Distribution of {key} Lengths (Correct vs Incorrect): {input_file.split("/")[-1].split(".")[0].split("_")[0]}')
    plt.xlabel(f'{key} Length (in tokens)')
    plt.ylabel('Count')
    
    # 显示图例 (此时图例里已经包含了 Avg 数值)
    plt.legend(loc='upper right') 
    
    plt.grid(axis='y', alpha=0.75)
    
    output_file = input_file.replace('.json', f'_length_dist_by_{key}_result.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    print(f"Correct Avg: {mean_correct:.2f}, Incorrect Avg: {mean_incorrect:.2f}")
    
# draw_acc_distribution(input_file)
draw_solution_length_distribution_by_result(input_solution_file, key='solution')