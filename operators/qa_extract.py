import os
import sys
# 添加父一级目录到 sys.path（上一级）
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC
from prompts.vqa import QAExtractPrompt
import os
import json
import tiktoken

from dataflow.core.prompt import prompt_restrict 

@prompt_restrict(QAExtractPrompt)
@OPERATOR_REGISTRY.register()
class QAExtractor(OperatorABC):
    def __init__(self,
                llm_serving: LLMServingABC = None,
                ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = QAExtractPrompt()
        
    def _convert_json(self, input_file, output_file):
        with open(input_file, 'r') as infile:
            data = list(json.load(infile))
        
        new_data = []
        
        id = 0
        for item in data:
            item['id'] = id
            item.pop('bbox', None)
            item.pop('page_idx', None)
            if item.get('type','') == 'list':
                if item['sub_type'] == 'text':
                    for idx, list_item in enumerate(item.get('list_items', [])):
                        new_item = {
                            'type': 'text',
                            'text': list_item,
                            'id': id + idx,
                        }
                        new_data.append(new_item)
                    id += len(item.get('list_items', []))
            else:
                new_data.append(item)
                id += 1
        
        with open(output_file, 'w') as outfile:
            json.dump(new_data, outfile, ensure_ascii=False)
            
    def _count_tokens(self, text: str) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    # def run(self, storage, input_json_paths: list[str], input_subject: str = "math") -> list[str]:
    #     dataframe = storage.read("dataframe")
    #     system_prompt = self.prompt_template.build_prompt(input_subject)
    #     user_inputs = []
    #     split_metadata = []  # 记录每个文件分了几个 chunk
    #     
    #     system_prompt_len = self._count_tokens(system_prompt)
    #
    #     for input_json_path in input_json_paths:
    #         converted_path = input_json_path.replace('.json', '_converted.json')
    #         self._convert_json(input_json_path, converted_path)
    #
    #         with open(converted_path, 'r') as infile:
    #             data = json.load(infile)
    #             assert isinstance(data, list), f"Expected list, got {type(data)} for {input_json_path}"
    #
    #         # === 分段处理 ===
    #         max_chunk_len = 128000  # 粗略 token 限制，可根据模型调整（不要开太大，不然模型效果会变差）
    #         current_chunk, current_len = [], system_prompt_len
    #         chunks = []
    #
    #         for item in data:
    #             text = json.dumps(item, ensure_ascii=False)
    #             item_len = self._count_tokens(text)
    #             # 若当前段超过限制，则换新chunk
    #             if current_len + item_len > max_chunk_len and current_chunk:
    #                 chunks.append(current_chunk)
    #                 current_chunk, current_len = [], 0
    #             current_chunk.append(item)
    #             current_len += item_len
    #
    #         if current_chunk:
    #             chunks.append(current_chunk)
    #
    #         # 记录每个 input_json 被分了几个 chunk
    #         split_metadata.append(len(chunks))
    #
    #         # 把每个 chunk 序列化后放入批量调用列表
    #         for chunk in chunks:
    #             user_inputs.append(json.dumps(chunk, ensure_ascii=False))
    #
    #     # === 批量生成 ===
    #     responses = self.llm_serving.generate_from_input(user_inputs, system_prompt)
    #
    #     # === 按 split_metadata 还原 ===
    #     recombined_responses = []
    #     idx = 0
    #     for num_chunks in split_metadata:
    #         merged_text = "\n".join(responses[idx: idx + num_chunks])
    #         recombined_responses.append(merged_text)
    #         idx += num_chunks
    #
    #     return recombined_responses

    def run(self, storage: DataFlowStorage, input_json_path_key: str = "json_path", input_subject_key: str = "subject", output_key: str = "extracted_qa") -> list:
        dataframe = storage.read("dataframe")
        # 获取输入数据
        if input_json_path_key not in dataframe.columns:
            raise ValueError(f"Column '{input_json_path_key}' not found in dataframe")
        
        if input_subject_key not in dataframe.columns:
            raise ValueError(f"Column '{input_subject_key}' not found in dataframe")
        
        json_paths = dataframe[input_json_path_key].tolist()
        subjects = dataframe[input_subject_key].tolist()
        
        user_inputs = []
        split_metadata = []  # 记录每个文件分了几个 chunk
        
        # 处理每个 JSON 文件
        for idx, input_json_path in enumerate(json_paths):
            subject = subjects[idx] if idx < len(subjects) else subjects[0] if subjects else "math"
            system_prompt = self.prompt_template.build_prompt(subject)
            system_prompt_len = self._count_tokens(system_prompt)
            
            converted_path = input_json_path.replace('.json', '_converted.json')
            self._convert_json(input_json_path, converted_path)
            with open(converted_path, 'r') as infile:
                data = json.load(infile)
                assert isinstance(data, list), f"Expected list, got {type(data)} for {input_json_path}"
            # === 分段处理 ===
            max_chunk_len = 128000  # 粗略 token 限制，可根据模型调整（不要开太大，不然模型效果会变差）
            current_chunk, current_len = [], system_prompt_len
            chunks = []

            for item in data:
                text = json.dumps(item, ensure_ascii=False)
                item_len = self._count_tokens(text)
                # 若当前段超过限制，则换新chunk
                if current_len + item_len > max_chunk_len and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk, current_len = [], system_prompt_len
                current_chunk.append(item)
                current_len += item_len

            if current_chunk:
                chunks.append(current_chunk)
            # 记录每个 input_json 被分了几个 chunk
            split_metadata.append(len(chunks))
            # 把每个 chunk 序列化后放入批量调用列表，同时记录对应的 system_prompt
            for chunk in chunks:
                user_inputs.append({
                    'user_input': json.dumps(chunk, ensure_ascii=False),
                    'system_prompt': system_prompt
                })

        # === 批量生成 ===
        # 按 system_prompt 分组批量调用，只要能正确还原顺序即可
        responses = [None] * len(user_inputs)
        current_batch = []
        current_batch_indices = []
        current_system_prompt = None
        
        for idx, item in enumerate(user_inputs):
            user_input = item['user_input']
            system_prompt = item['system_prompt']
            
            if current_system_prompt is None:
                current_system_prompt = system_prompt
                current_batch = [user_input]
                current_batch_indices = [idx]
            elif system_prompt == current_system_prompt:
                current_batch.append(user_input)
                current_batch_indices.append(idx)
            else:
                # 处理当前批次
                batch_responses = self.llm_serving.generate_from_input(user_inputs=current_batch, system_prompt=current_system_prompt)
                for batch_idx, resp in zip(current_batch_indices, batch_responses):
                    responses[batch_idx] = resp
                # 开始新批次
                current_system_prompt = system_prompt
                current_batch = [user_input]
                current_batch_indices = [idx]
        
        # 处理最后一批
        if current_batch:
            batch_responses = self.llm_serving.generate_from_input(user_inputs=current_batch, system_prompt=current_system_prompt)
            for batch_idx, resp in zip(current_batch_indices, batch_responses):
                responses[batch_idx] = resp

        # === 按 split_metadata 还原 ===
        recombined_responses = []
        idx = 0
        for num_chunks in split_metadata:
            merged_text = "\n".join(responses[idx: idx + num_chunks])
            recombined_responses.append(merged_text)
            idx += num_chunks

        # 将结果写入 dataframe
        dataframe[output_key] = recombined_responses
        output_file = storage.write(dataframe)
        self.logger.info(f"Extracted QA results saved to {output_file}")
        
        return [output_key,]