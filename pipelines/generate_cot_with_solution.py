import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataflow.operators.core_text import PandasOperator, PromptTemplatedGenerator
from operators.bench_evaluate import BenchDatasetEvaluatorQuestion
from operators.vqa_answer_generator import VQAReasoningAnswerGenerator

from dataflow.serving import APILLMServing_request, APIVLMServing_openai, LocalVLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.operators.reasoning import (
    ReasoningAnswerGenerator,
    ReasoningAnswerGroundTruthFilter
)
from prompts.cot_with_solution import GeneralAnswerWithSolutionGeneratorPrompt
from dataflow.operators.core_text import GeneralFilter
from dataflow import get_logger

from typing import Iterable
import re

from dataflow.pipeline import PipelineABC

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

class CotWithSolutionPipeline(PipelineABC):
    def __init__(self, first_entry_file_name, cache_path, file_name_prefix, eval_result_path, cache_type="json"):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name, ### 现在dataflow还不支持图架构，难以把qa，qas，qs的分类放进来，这个算法在 /data1/hzh/vqa/completeness_filter.py
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        
        self.logger = get_logger()

        self.llm_answer_serving = LocalVLMServing_vllm(
            hf_model_name_or_path="/data0/models/Qwen3-VL-32B-Thinking",
            vllm_temperature=0.0,
            vllm_tensor_parallel_size=8,
            vllm_max_tokens=8192,
            vllm_max_model_len=12800,
            vllm_gpu_memory_utilization=0.7,
            vllm_limit_mm_per_prompt=10,
            vllm_repetition_penalty=1.1,
            batch_size=128
        )
        
        self.llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-5-mini",
                max_workers=100,
        )
        
        # 难度过滤
        self.difficulty_filter = GeneralFilter(
            filter_rules=[lambda df: df['accuracy'] <= 1.0]
        )
        
        #llm 回答
        self.answer_generator = VQAReasoningAnswerGenerator(
            llm_serving=self.llm_answer_serving,
            prompt_template=GeneralAnswerWithSolutionGeneratorPrompt(),
            skip_text_only=True,
        )
        
        self.solution_filter = GeneralFilter(
            filter_rules=[lambda df: df['solution'] != ""]
        )
        
        self.think_cleaner = PandasOperator(process_fn=[ make_remove_think_fn(input_key="generated_cot", output_key="generated_cot") ])
        
        ## llm核对
        self.answer_groundtruth_filter = BenchDatasetEvaluatorQuestion(
            compare_method="semantic",
            llm_serving=self.llm_serving,
            prompt_template=None, # using default prompt
            eval_result_path=eval_result_path,
            support_subquestions=True,
            skip_true=True
        )
        
        
    def forward(self):
                
        # self.difficulty_filter.run(storage = self.storage.step())
        
        # llm回答
        self.answer_generator.run(
            storage = self.storage.step(),
            input_key = "question", 
            output_key = "generated_cot",
            input_caption_key="captions",
            input_solution_key="solution",
        )
        
        self.solution_filter.run(storage = self.storage.step())
        
        self.think_cleaner.run(storage = self.storage.step())
        
        self.answer_groundtruth_filter.run(
            storage=self.storage.step(), 
            input_test_answer_key="generated_cot",
            input_gt_answer_key="answer",
            input_question_key="question",
        )
            
if __name__ == "__main__":
    first_entry_file_name=f"/data0/djw/VQA_1209/caption_cache/gpt-5-mini_step3.json"
    cache_path = f"./cot_cache/vqa_1209_top1000"
    file_name_prefix = f"qwen3_vl_32b_cot_with_solution"
    eval_result_path = f"./cot_cache/all_vqa_only/eval_results.jsonl"
    
    model = CotWithSolutionPipeline(first_entry_file_name, cache_path, file_name_prefix, eval_result_path)
    model.compile()
    model._compiled_forward(resume_step=1)
