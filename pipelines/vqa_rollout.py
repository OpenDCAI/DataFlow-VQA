import os
import sys
import json
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataflow.operators.core_text import PandasOperator, PromptTemplatedGenerator
from operators.bench_evaluate import BenchDatasetEvaluatorQuestion
from operators.vqa_answer_generator import VQAReasoningAnswerGenerator

from dataflow.serving import APILLMServing_request, LocalVLMServing_vllm
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

def make_rollout_dup_fn(num: int):
    def fn(df):
        base = df.copy().reset_index(drop=True)

        parts = []
        for i in range(num):
            part = base.copy()
            part["rollout_index"] = i
            
            parts.append(part)

        out = pd.concat(parts, ignore_index=True)
        return out

    return fn


def make_rollout_aggregate_fn():
    def fn(df):
        # 规范化 answer_match_result → bool
        s = df["answer_match_result"]
        if s.dtype != bool:
            s = s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
        df = df.copy()
        df["_match"] = s

        group = df.groupby("question", sort=False)

        agg = group["_match"].agg(
            rollout_num="size",
            accuracy="mean"
        )

        # 每题的第一条记录保留其他字段
        firsts = group.first()
        if "_match" in firsts.columns:
            firsts = firsts.drop(columns=["_match"])

        result = firsts.join(agg).reset_index()

        return result

    return fn

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

            df["llm_answer"] = df["llm_answer"].apply(clean_text)

        return df

    return fn


class BenchSamplingPipeline():
    def __init__(self, first_entry_file_name, cache_path, file_name_prefix, cache_type="json"):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name, ### 现在dataflow还不支持图架构，难以把qa，qas，qs的分类放进来，这个算法在 /data1/hzh/vqa/completeness_filter.py
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        self.llm_answer_serving = LocalVLMServing_vllm(
            hf_model_name_or_path="/data0/models/Qwen3-VL-8B-Thinking",
            vllm_temperature=0.5,
            vllm_tensor_parallel_size=4,
            vllm_max_tokens=8192,
            vllm_max_model_len=128000,
            vllm_gpu_memory_utilization=0.6,
            vllm_limit_mm_per_prompt=10,
            vllm_repetition_penalty=1.1,
            batch_size=128
        )
        
        self.llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-5-mini",
                max_workers=100,
        )
        
        self.dup_operator = PandasOperator(process_fn=[ make_rollout_dup_fn(32) ])
        self.agg_operator = PandasOperator(process_fn=[ make_rollout_aggregate_fn() ])
        
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
            eval_result_path="./rollout_cache/math-test/eval_result_math_test.json",
            support_subquestions=True
        )
        
    def forward(self, input_image_default_basedir):
                
        self.dup_operator.run(
            storage = self.storage.step(),
        )
        
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
                input_test_answer_key="llm_answer",
                input_gt_answer_key="answer",
                input_question_key="question",
            )
            self.agg_operator.run(storage=self.storage.step())
        except:
            pass

if __name__ == "__main__":
    
    first_entry_file_name=f"/data1/VQA_ready_data/all_vqa.jsonl"
    cache_path = f"./rollout_cache/all_math"
    file_name_prefix = f"math-Qwen3-8B-Thinking"
    input_image_default_basedir = f"./"
    
    model = BenchSamplingPipeline(first_entry_file_name, cache_path, file_name_prefix)
    model.forward(input_image_default_basedir)