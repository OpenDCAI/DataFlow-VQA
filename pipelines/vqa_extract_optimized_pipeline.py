import os
import sys
# 添加父一级目录到 sys.path（上一级）
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from operators.vqa_extractor import VQAExtractor

class VQA_extract_optimized_pipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./examples/VQA/vqa_extract_interleaved_test.jsonl",
            cache_path="./vqa_extract_optimized_cache",
            file_name_prefix="vqa",
            cache_type="jsonl",
        )
        
        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            key_name_of_api_key="DF_API_KEY",
            model_name="gemini-2.5-pro",
            max_workers=100,
        )
        
        self.vqa_extractor = VQAExtractor(
            llm_serving=self.llm_serving
        )
        
    def forward(self):
        # 单一算子：包含预处理、QA提取、后处理的所有功能
        self.vqa_extractor.run(
            storage=self.storage.step(),
            question_pdf_path_key="question_pdf_path",
            answer_pdf_path_key="answer_pdf_path",
            pdf_path_key="pdf_path",  # 支持 interleaved 模式
            subject_key="subject",
            output_dir_key="output_dir",
            output_jsonl_key="output_jsonl_path",
            mineru_backend='vlm-vllm-engine',
        )



if __name__ == "__main__":
    # jsonl中每一行包含question_pdf_path, answer_pdf_path, subject (math, physics, chemistry, ...), output_dir
    # 如果question和answer在同一份pdf中，请将question_pdf_path和answer_pdf_path设置为相同的路径，会自动切换为interleaved模式
    pipeline = VQA_extract_optimized_pipeline()
    pipeline.forward()