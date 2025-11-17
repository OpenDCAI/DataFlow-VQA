import json
import os
import re
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from utils.format_utils import merge_qa_pair, jsonl_to_md

@OPERATOR_REGISTRY.register()
class VQAMerger(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
    
    def run(self, storage: DataFlowStorage, input_qa_list_key: str = "qa_list",
            output_dir_key: str = "output_dir", mode_key: str = "mode",
            interleaved_key: str = "interleaved", output_root_key: str = "output_root",
            output_jsonl_key: str = "output_jsonl_path") -> list:
        dataframe = storage.read("dataframe")
        
        if input_qa_list_key not in dataframe.columns:
            raise ValueError(f"Column '{input_qa_list_key}' not found in dataframe")
        
        qa_lists = dataframe[input_qa_list_key].tolist()
        output_dirs = dataframe[output_dir_key].tolist() if output_dir_key in dataframe.columns else [None] * len(dataframe)
        modes = dataframe[mode_key].tolist() if mode_key in dataframe.columns else ["question"] * len(dataframe)
        interleaved_list = dataframe[interleaved_key].tolist() if interleaved_key in dataframe.columns else [False] * len(dataframe)
        output_roots = dataframe[output_root_key].tolist() if output_root_key in dataframe.columns else [None] * len(dataframe)
        
        # 按 output_root 分组处理
        output_groups = {}
        for idx, (qa_list, output_dir, mode, interleaved, output_root) in enumerate(zip(qa_lists, output_dirs, modes, interleaved_list, output_roots)):
            if output_root is None:
                # 如果没有 output_root，从 output_dir 推断
                if output_dir:
                    if mode in output_dir:
                        output_root = os.path.dirname(output_dir)
                    else:
                        output_root = output_dir
                else:
                    continue
            
            if output_root not in output_groups:
                output_groups[output_root] = {
                    "question": None,
                    "answer": None,
                    "interleaved": interleaved
                }
            
            if mode == "question":
                output_groups[output_root]["question"] = (qa_list, output_dir)
            elif mode == "answer":
                output_groups[output_root]["answer"] = (qa_list, output_dir)
        
        output_jsonl_paths = []
        
        for output_root, group_info in output_groups.items():
            q_qa_list, q_output_dir = group_info["question"] if group_info["question"] else (None, None)
            a_qa_list, a_output_dir = group_info["answer"] if group_info["answer"] else (None, None)
            interleaved = group_info["interleaved"]
            
            # 写入 question jsonl
            q_jsonl_path = os.path.join(output_root, "vqa_extracted_questions.jsonl")
            if q_qa_list:
                with open(q_jsonl_path, 'w', encoding='utf-8') as f:
                    for item in q_qa_list:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 写入 answer jsonl（如果不是 interleaved）
            a_jsonl_path = None
            if not interleaved and a_qa_list:
                a_jsonl_path = os.path.join(output_root, "vqa_extracted_answers.jsonl")
                with open(a_jsonl_path, 'w', encoding='utf-8') as f:
                    for item in a_qa_list:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 合并
            merged_jsonl = os.path.join(output_root, "vqa_merged_qa_pairs.jsonl")
            if not interleaved and a_jsonl_path:
                merge_qa_pair(q_jsonl_path, a_jsonl_path, merged_jsonl)
            else:
                os.system(f"cp {q_jsonl_path} {merged_jsonl}")
            
            # 过滤
            filtered_items = []
            total_count = 0
            with open(merged_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    total_count += 1
                    item = json.loads(line)
                    if item.get('question','').strip() and (item.get('answer','').strip() or item.get('solution','').strip()):
                        filtered_items.append(item)
            
            self.logger.info(f"Before filter: {total_count}, After filter: {len(filtered_items)}")
            
            filtered_jsonl = os.path.join(output_root, "vqa_filtered_qa_pairs.jsonl")
            with open(filtered_jsonl, 'w', encoding='utf-8') as f:
                for item in filtered_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 转换为 markdown
            md_output = os.path.join(output_root, "vqa_filtered_qa_pairs.md")
            jsonl_to_md(filtered_jsonl, md_output)
            
            output_jsonl_paths.append(filtered_jsonl)
        
        # 为每个 dataframe 行分配对应的 output_jsonl_path
        result_paths = []
        for idx, (output_dir, output_root) in enumerate(zip(output_dirs, output_roots)):
            if output_root is None:
                if output_dir:
                    if modes[idx] in output_dir:
                        output_root = os.path.dirname(output_dir)
                    else:
                        output_root = output_dir
                else:
                    result_paths.append(None)
                    continue
            result_paths.append(os.path.join(output_root, "vqa_filtered_qa_pairs.jsonl"))
        
        dataframe[output_jsonl_key] = result_paths
        output_file = storage.write(dataframe)
        self.logger.info(f"Merged QA pairs saved to {output_file}")
        
        return [output_jsonl_key,]

