from dataflow.operators.knowledge_cleaning import FileOrURLToMarkdownConverterAPI

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from operators.pdf2vqa import MinerU2LLMInputOperator, LLMOutputParser, QA_Merger, PDF_Merger
from dataflow.operators.core_text import ChunkedPromptedGenerator

from dataflow.pipeline import PipelineABC
from dataflow.prompts.pdf2vqa import QAExtractPrompt

from pypdf import PdfWriter

import os
import json
import re
import argparse

#######CONFIGS##########
API_URL = "https://api.vectortara.com/v1/chat/completions"
MODEL_NAME = "gemini-2.5-pro"
MAX_WORKERS = 100
########################
    
class PDF_VQA_extract_optimized_pipeline(PipelineABC):
    def __init__(self, input_file):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path="./cache",
            file_name_prefix="vqa",
            cache_type="jsonl",
        )
        
        self.llm_serving = APILLMServing_request(
            api_url=API_URL,
            key_name_of_api_key="DF_API_KEY",
            model_name=MODEL_NAME,
            max_workers=MAX_WORKERS,
        )
        
        self.vqa_extract_prompt = QAExtractPrompt()
        
        self.pdf_merger = PDF_Merger(output_dir="./cache")
        
        self.mineru_executor = FileOrURLToMarkdownConverterAPI(intermediate_dir = "intermediate")

        self.input_formatter = MinerU2LLMInputOperator()
        self.vqa_extractor = ChunkedPromptedGenerator(
            llm_serving=self.llm_serving,
            system_prompt = self.vqa_extract_prompt.build_prompt(),
            max_chunk_len=128000,
        )
        self.llm_output_parser = LLMOutputParser(output_dir="./cache", intermediate_dir="intermediate")
        self.qa_merger = QA_Merger(output_dir="./cache", strict_title_match=False)


    def forward(self):
        self.pdf_merger.run(
            storage=self.storage.step(),
            input_pdf_list_key="input_pdf_paths",
            input_name_key="name",
            output_pdf_path_key="merged_pdf_path",
        )
        self.mineru_executor.run(
            storage=self.storage.step(),
            input_key="merged_pdf_path",
            output_key="vqa_markdown_path",
        )
        self.input_formatter.run(
            storage=self.storage.step(),
            input_markdown_path_key="vqa_markdown_path",
            output_converted_layout_key="converted_vqa_layout_path",
        )
        self.vqa_extractor.run(
            storage=self.storage.step(),
            input_path_key="converted_vqa_layout_path",
            output_path_key="extracted_llm_vqa_path",
        )
        self.llm_output_parser.run(
            storage=self.storage.step(),
            input_response_path_key="extracted_llm_vqa_path",
            input_converted_layout_path_key="converted_vqa_layout_path",
            input_name_key="name",
            output_qalist_path_key="extracted_vqa_path",
        )
        self.qa_merger.run(
            storage=self.storage.step(),
            input_qalist_path_key="extracted_vqa_path",
            input_name_key="name",
            output_merged_qalist_path_key="output_merged_vqalist_path",
            output_merged_md_path_key="output_merged_md_path",
            output_qa_item_key="vqa_pair",
        )



if __name__ == "__main__":
    #是用命令行读入输入文件，输出文件夹
    parser = argparse.ArgumentParser(description="Run PDF VQA Extract Optimized Pipeline")
    parser.add_argument("--input_file", type=str, default="./examples/VQA/vqa_extract_test.jsonl", help="Path to the input JSONL file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the output files")
    args = parser.parse_args()
    
    
    # jsonl中每一行包含input_pdf_path, name (math1, math2, physics1, chemistry1, ...)
    # pipeline = PDF_VQA_extract_optimized_pipeline(input_file=args.input_file)
    # pipeline.compile()
    # pipeline.forward()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 首先找到cache中最大的一个vqa_step*.jsonl文件
    cache_files = os.listdir("./cache")
    step_files = [f for f in cache_files if re.match(r"vqa_step\d+\.jsonl", f)]
    step_numbers = [int(re.findall(r"vqa_step(\d+)\.jsonl", f)[0]) for f in step_files]
    max_step = max(step_numbers)
    max_step_file = f"./cache/vqa_step{max_step}.jsonl"
    
    # 从这个文件中读取每一行，获取output_qa_item_key的内容，并保存到output_dir中的raw_vqa.jsonl文件中
    output_qa_item_key = "vqa_pair"
    with open(max_step_file, "r") as f_in, open(os.path.join(output_dir, "raw_vqa.jsonl"), "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            qa_item = data[output_qa_item_key]
            name = data["name"]
            output_data = {"name": name, **qa_item, "image_basedir": os.path.abspath(output_dir)}
            if not output_data["solution"]:
                output_data["solution"] = output_data["answer"]
            f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            
            # 把name对应的文件夹也复制过来
            src_dir = os.path.join("cache", name)
            if os.path.exists(src_dir):
                os.system(f"cp -r {src_dir} {output_dir}")