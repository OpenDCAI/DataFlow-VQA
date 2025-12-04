import os
import sys
import json
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataflow.operators.core_text import PandasOperator, PromptTemplatedGenerator
from operators.bench_evaluate import BenchDatasetEvaluatorQuestion
from operators.vqa_answer_generator import VQAReasoningAnswerGenerator

from dataflow.serving import APILLMServing_request, APIVLMServing_openai
from dataflow.utils.storage import FileStorage
from dataflow.operators.reasoning import (
    ReasoningAnswerGenerator,
    ReasoningAnswerGroundTruthFilter
)
from dataflow.prompts.reasoning.math import MathAnswerGeneratorPrompt
from prompts.bench_sampling import BenchSamplingPrompt, SubQuestionSplitingPrompt, QAFilterPrompt
from dataflow.prompts.core_text import StrFormatPrompt
from dataflow.operators.core_text import GeneralFilter

from typing import Iterable
import re

def make_remove_think_fn():
    pattern = re.compile(r'<think>.*?</think>', flags=re.DOTALL | re.IGNORECASE)
    def fn(df):
        df = df.copy()
        if "llm_answer" in df.columns:
            def clean_text(t):
                if pd.isna(t):
                    return t
                if "</think>" not in t:
                    return t.strip()
                s = "<think>" + str(t)
                return pattern.sub("", s).strip()

            df["llm_short_answer"] = df["llm_answer"].apply(clean_text)

        return df
    
    return fn

class BenchSamplingPipeline():
    def __init__(self, first_entry_file_name, cache_path, file_name_prefix, eval_result_path, cache_type="json"):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name, ### 现在dataflow还不支持图架构，难以把qa，qas，qs的分类放进来，这个算法在 /data1/hzh/vqa/completeness_filter.py
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        self.llm_answer_serving = APIVLMServing_openai(
                api_url="http://123.129.219.111:3000/v1/",
                model_name="gpt-5-mini",
                max_workers=100,
                timeout=120
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
            prompt_template=MathAnswerGeneratorPrompt(),
            skip_text_only=True,
        )
        
        self.think_cleaner = PandasOperator(process_fn=[ make_remove_think_fn() ])
        
        ## llm核对
        self.answer_groundtruth_filter = BenchDatasetEvaluatorQuestion(
            compare_method="semantic",
            llm_serving=self.llm_serving,
            prompt_template=None, # using default prompt
            eval_result_path=eval_result_path,
            support_subquestions=True
        )
        
    def forward(self, input_image_default_basedir):
                
        self.difficulty_filter.run(storage = self.storage.step())
        
        # llm回答
        self.answer_generator.run(
            storage = self.storage.step(),
            input_key = "question", 
            output_key = "llm_answer",
            input_image_default_basedir=input_image_default_basedir
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
        except:
            pass

if __name__ == "__main__":
    first_entry_file_name=f"/data1/djw/DataFlow-VQA/rollout_cache/all_math_vqa_only/math-Qwen3-8B-Instruct_step5.json"
    cache_path = f"./eval_cache/all_math_vqa_only"
    file_name_prefix = f"math-gpt-5-mini"
    eval_result_path = f"./eval_cache/all_math_vqa_only/eval_results.jsonl"
    input_image_default_basedir = f"./"
    
    model = BenchSamplingPipeline(first_entry_file_name, cache_path, file_name_prefix, eval_result_path)
    model.forward(input_image_default_basedir)