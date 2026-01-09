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
from dataflow.prompts.reasoning.general import GeneralAnswerGeneratorPrompt
from dataflow.operators.core_text import GeneralFilter
from dataflow import get_logger

from typing import Iterable
import re

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

class RejectSamplingPipeline():
    def __init__(self, first_entry_file_name, cache_path, file_name_prefix, eval_result_path, max_retries, cache_type="json"):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name, ### 现在dataflow还不支持图架构，难以把qa，qas，qs的分类放进来，这个算法在 /data1/hzh/vqa/completeness_filter.py
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        
        self.max_retries = max_retries
        self.logger = get_logger()

        self.llm_answer_serving = LocalVLMServing_vllm(
            hf_model_name_or_path="/data0/models/Qwen3-VL-32B-Thinking",
            vllm_temperature=0.7,
            vllm_tensor_parallel_size=4,
            vllm_max_tokens=8192,
            vllm_max_model_len=12800,
            vllm_gpu_memory_utilization=0.7,
            vllm_limit_mm_per_prompt=15,
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
            prompt_template=GeneralAnswerGeneratorPrompt(),
            skip_text_only=False,
            input_image_default_basedir="/data0/djw/camelai"
        )
        
        self.think_cleaner = PandasOperator(process_fn=[ make_remove_think_fn(input_key="generated_cot", output_key="llm_short_answer") ])
        
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
        
        for i in range(self.max_retries):
        
            input_skip_key="answer_match_result" if i > 0 else None
            
            # llm回答（跳过已经做对的题）
            self.answer_generator.run(
                storage = self.storage.step(),
                input_key = "question", 
                output_key = "generated_cot",
                # input_caption_key="captions",
                input_skip_key=input_skip_key,
            )
            

            # 有可能会有空的文件，所以要try
            try:
                self.think_cleaner.run(storage = self.storage.step())
                # TODO:
                # 这个judge很有问题，很不准确，得改，可以考虑sympy?
                self.answer_groundtruth_filter.run(
                    storage=self.storage.step(), 
                    input_test_answer_key="llm_short_answer",
                    input_gt_answer_key="answer",
                    input_question_key="question",
                )
                self.storage.operator_step += 1
                correct_num = self.storage.read(output_type="dataframe")["answer_match_result"].sum()
                self.logger.info(f"After reject sampling round {i+1}, correct_num: {correct_num}")
                self.storage.operator_step -= 1
            except Exception as e:
                self.logger.warning(f"Eval failed at reject sampling round {i+1}: {e}")

if __name__ == "__main__":
    first_entry_file_name=f"/data0/djw/camelai/cleaned.json"
    cache_path = f"./cot_cache/camelai_math/"
    file_name_prefix = f"math-Qwen3-32B-Thinking"
    eval_result_path = f"./cot_cache/camelai_math/eval_results.jsonl"
    max_retries = 3
    
    model = RejectSamplingPipeline(first_entry_file_name, cache_path, file_name_prefix, eval_result_path, max_retries)
    model.forward()
