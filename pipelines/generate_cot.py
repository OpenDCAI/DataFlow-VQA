import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataflow.operators.core_text import PandasOperator
from operators.bench_evaluate import BenchDatasetEvaluatorQuestion
from operators.vqa_answer_generator import VQAReasoningAnswerGenerator

from dataflow.serving import APILLMServing_request, APIVLMServing_openai, LocalVLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.operators.reasoning import (
    ReasoningAnswerGenerator,
    ReasoningAnswerGroundTruthFilter
)
from dataflow.prompts.reasoning.math import MathAnswerGeneratorPrompt
from dataflow.operators.core_text import GeneralFilter
from dataflow import get_logger
from dataflow.pipeline import PipelineABC

from typing import Iterable
import re
import argparse
import shutil

#######CONFIGS##########
ANSWER_API_URL = "https://api.vectortara.com/v1"
ANSWER_MODEL_NAME = "gpt-5-mini"
JUDGE_API_URL = "https://api.vectortara.com/v1/chat/completions"
JUDGE_MODEL_NAME = "gpt-5-mini"
MAX_WORKERS = 100
########################

def make_remove_think_fn(input_key, output_key):
    pattern = re.compile(r'<think>.*?</think>', flags=re.DOTALL | re.IGNORECASE)
    def fn(df):
        df = df.copy()
        if input_key in df.columns:
            def clean_text(t):
                if pd.isna(t):
                    return t
                if "</think>" not in t:
                    return t.strip()
                s = "<think>" + str(t)
                return pattern.sub("", s).strip()

            df[output_key] = df[input_key].apply(clean_text)

        return df
    
    return fn

class RejectSamplingPipeline(PipelineABC):
    def __init__(self, first_entry_file_name, max_retries=5):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path="./cot_cache",
            file_name_prefix="reject_sampling",
            cache_type="jsonl",
        )
        
        self.max_retries = max_retries
        self.logger = get_logger()

        self.llm_answer_serving = APIVLMServing_openai(
                api_url=ANSWER_API_URL,
                model_name=ANSWER_MODEL_NAME,
                max_workers=MAX_WORKERS,
                timeout=600.0,
                max_tokens=8192,
                temperature=0.7,
        )
        
        self.llm_serving = APILLMServing_request(
                api_url=JUDGE_API_URL,
                model_name=JUDGE_MODEL_NAME,
                max_workers=100,
                read_timeout=300.0
        )
        
        # 难度过滤
        self.difficulty_filter = GeneralFilter(
            filter_rules=[lambda df: df['accuracy'] <= 1.0]
        )
        
        #llm 回答
        self.answer_generator = VQAReasoningAnswerGenerator(
            llm_serving=self.llm_answer_serving,
            prompt_template=MathAnswerGeneratorPrompt(),
            skip_text_only=False,
        )
        
        self.think_cleaner = PandasOperator(process_fn=[ make_remove_think_fn(input_key="generated_cot", output_key="llm_short_answer") ])
        
        self.noop = PandasOperator(process_fn=[ lambda df: df ])
        
        ## llm核对
        self.answer_groundtruth_filter = BenchDatasetEvaluatorQuestion(
            compare_method="semantic",
            llm_serving=self.llm_serving,
            prompt_template=None, # using default prompt
            eval_result_path="./cot_cache/eval_results.jsonl",
            support_subquestions=True,
            skip_true=True
        )
        
    def forward(self):
        self.noop.run(storage = self.storage.step(), output_key="answer_match_result") # for pipeline compliation, do nothing
        for i in range(self.max_retries):
        
            input_skip_key="answer_match_result" if i > 0 else None
            
            # llm回答（跳过已经做对的题）
            self.answer_generator.run(
                storage = self.storage.step(),
                input_key = "question", 
                output_key = "generated_cot",
                input_skip_key=input_skip_key,
                input_image_basedir_key="image_basedir",
            )
            

            self.think_cleaner.run(storage = self.storage.step(), output_key="llm_short_answer")

            self.answer_groundtruth_filter.run(
                storage=self.storage.step(), 
                input_test_answer_key="llm_short_answer",
                input_gt_answer_key="answer",
                input_question_key="question",
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Curation Pipeline")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of reject sampling rounds")
    args = parser.parse_args()
    
    
    first_entry_file_name=args.input_file
    max_retries = args.max_retries
    
    model = RejectSamplingPipeline(args.input_file, args.max_retries)
    model.compile()
    model.forward()
    
    # 首先找到cache中最大的一个reject_sampling_data_step*.jsonl文件
    cache_files = os.listdir("./cot_cache")
    step_files = [f for f in cache_files if re.match(r"reject_sampling_step\d+\.jsonl", f)]
    step_numbers = [int(re.findall(r"reject_sampling_step(\d+)\.jsonl", f)[0]) for f in step_files]
    max_step = max(step_numbers)
    max_step_file = f"./cot_cache/reject_sampling_step{max_step}.jsonl"
    
    # 将该文件复制到output目录下,并命名为curated_data.jsonl
    # output目录应当与input_file同级，否则图片路径会有问题
    output_dir = os.path.dirname(args.input_file)
    output_file = os.path.join(output_dir, "curated_vqa_with_cot.jsonl")
    shutil.copy(max_step_file, output_file)
    print(f"Curated data with cot saved to: {output_file}")