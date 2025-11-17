import os
import sys
# 添加父一级目录到 sys.path（上一级）
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from operators.qa_extract import QAExtractor
from operators.vqa_task_expander import VQATaskExpander
from operators.vqa_layout_extractor import VQALayoutExtractor
from operators.vqa_response_converter import VQAResponseConverter
from operators.vqa_merger import VQAMerger

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
        
        self.task_expander = VQATaskExpander()
        self.layout_extractor = VQALayoutExtractor(mineru_backend='vlm-vllm-engine')
        self.qa_extractor = QAExtractor(llm_serving=self.llm_serving)
        self.response_converter = VQAResponseConverter()
        self.merger = VQAMerger()
        
    def forward(self):
        # Stage 1: 扩展任务（将输入扩展为 question 和 answer 任务）
        self.task_expander.run(
            storage=self.storage.step(),
            question_pdf_path_key="question_pdf_path",
            answer_pdf_path_key="answer_pdf_path",
            pdf_path_key="pdf_path",  # 支持 interleaved 模式
            subject_key="subject",
            output_dir_key="output_dir",
        )
        
        # Stage 2: Layout 提取
        self.layout_extractor.run(
            storage=self.storage.step(),
            input_pdf_path_key="pdf_path",
            output_dir_key="output_dir",
            output_json_path_key="json_path",
            mode_key="mode",
        )
        
        # Stage 3: QA 提取
        self.qa_extractor.run(
            storage=self.storage.step(),
            input_json_path_key="json_path",
            input_subject_key="subject",
            output_key="extracted_qa",
        )
        
        # Stage 4: Response 转换
        self.response_converter.run(
            storage=self.storage.step(),
            input_response_key="extracted_qa",
            input_json_path_key="json_path",
            output_dir_key="output_dir",
            mode_key="mode",
            output_qa_list_key="qa_list",
        )
        
        # Stage 5: 合并和过滤
        self.merger.run(
            storage=self.storage.step(),
            input_qa_list_key="qa_list",
            output_dir_key="output_dir",
            mode_key="mode",
            interleaved_key="interleaved",
            output_root_key="output_root",
            output_jsonl_key="output_jsonl_path",
        )



if __name__ == "__main__":
    # jsonl中每一行包含question_pdf_path, answer_pdf_path, subject (math, physics, chemistry, ...), output_dir
    # 如果question和answer在同一份pdf中，请将question_pdf_path和answer_pdf_path设置为相同的路径，会自动切换为interleaved模式
    pipeline = VQA_extract_optimized_pipeline()
    pipeline.forward()