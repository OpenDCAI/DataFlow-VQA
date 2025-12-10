import json
import re
import argparse
import sys

def process_jsonl(input_file, output_file):
    # 定义正则表达式模式和替换目标
    # 模式：匹配 /data1/VQA_ready_data
    pattern = r"/data1/VQA_ready_data"
    # 替换为：/jizhicfs/herunming/vqa_wzh/images
    replacement = r"/jizhicfs/herunming/vqa_wzh/images"
    
    modified_count = 0
    line_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            print(f"正在处理文件: {input_file} ...")
            
            for line in f_in:
                line_count += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_count} 行不是有效的 JSON，已跳过。")
                    continue

                # 检查是否存在 image_basedir 字段
                if "image_basedir" in data and isinstance(data["image_basedir"], str):
                    original_path = data["image_basedir"]
                    
                    # 使用正则表达式进行替换
                    new_path = re.sub(pattern, replacement, original_path)
                    
                    # 如果发生了替换，更新数据并计数
                    if new_path != original_path:
                        data["image_basedir"] = new_path
                        modified_count += 1
                
                # 将处理后的数据写回新文件 (ensure_ascii=False 保证中文不乱码)
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"处理完成！")
        print(f"总行数: {line_count}")
        print(f"修改行数: {modified_count}")
        print(f"输出文件: {output_file}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_file}'")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="批量替换JSONL文件中 image_basedir 字段的路径")
    parser.add_argument("input", help="输入的 jsonl 文件路径")
    parser.add_argument("output", help="输出的 jsonl 文件路径")
    
    args = parser.parse_args()
    
    process_jsonl(args.input, args.output)
