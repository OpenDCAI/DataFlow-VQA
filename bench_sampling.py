import os
import sys
import json
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataflow.operators.core_text import PandasOperator, PromptTemplatedGenerator
from operators.bench_evaluate import BenchDatasetEvaluatorQuestion

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.reasoning import (
    ReasoningAnswerGenerator,
    ReasoningAnswerGroundTruthFilter
)
from dataflow.prompts.reasoning.math import MathAnswerGeneratorPrompt
from prompts.bench_sampling import BenchSamplingPrompt, SubQuestionSplitingPrompt, QAFilterPrompt
from dataflow.prompts.core_text import StrFormatPrompt
from dataflow.operators.core_text import GeneralFilter

class BenchSamplingPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./examples/VQA/vqa_filtered_qa_pairs.jsonl", ### 现在dataflow还不支持图架构，难以把qa，qas，qs的分类放进来，这个算法在 /data1/hzh/vqa/completeness_filter.py
            cache_path="./cache",
            file_name_prefix="math_test",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-5-mini",
                max_workers=100,
        )
        
        self.llm_answer_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-5-mini",
                max_workers=100,
        )
        #拆小题
        self.sub_qa_justify = PromptTemplatedGenerator(
            llm_serving = self.llm_serving,
            prompt_template = SubQuestionSplitingPrompt()
        )
        self.sub_qa_spliter = PandasOperator(
            [split_generated_content]
        )
        
        # 判断题型
        self.question_type_justify = PromptTemplatedGenerator(
            llm_serving = self.llm_serving,
            prompt_template = BenchSamplingPrompt()
        )
        self.completeness_filter = PandasOperator(
            [extract_type_and_reason]
        )
        
        # TODO:
        # answer + 题目过滤：题目或answer有那种根据lemma x.x的不具体描述、答案不完整这种、"Give an example"这种可以给无穷多答案的问题
        # answer 太长，看有没有必要refined变精简，提升llm as judge的性能。
        
        self.qa_filter = PromptTemplatedGenerator(
            llm_serving = self.llm_serving,
            prompt_template = QAFilterPrompt()
        )
        self.qa_filter_processor = PandasOperator(
            [extract_filter_result_and_reason]
        )
        self.qa_filter_executor = GeneralFilter(
            filter_rules=[lambda df: df['filter_result'] == 'true']
        )
        
        #llm 回答
        self.answer_generator = ReasoningAnswerGenerator(
            llm_serving=self.llm_answer_serving,
            prompt_template=MathAnswerGeneratorPrompt()
        )
        
        ## llm核对
        self.answer_groundtruth_filter = BenchDatasetEvaluatorQuestion(
            compare_method="semantic",
            llm_serving=self.llm_serving,
            prompt_template=None, # using default prompt
            eval_result_path="../eval_result_math_test.json",
            support_subquestions=True
        )
        
    def forward(self):
        # self.sub_qa_justify.run(
        #     storage = self.storage.step(),
        #     output_key = "split_qa",
        #     input_question  = "question",
        #     input_answer = "answer",
        # )
        
        # self.sub_qa_spliter.run(
        #     storage = self.storage.step(),
        # )
        
        # self.question_type_justify.run(
        #     storage = self.storage.step(),
        #     input_question = "question",
        #     input_answer = "answer",
        #     output_key = "question_type"
        # )
        
        # self.completeness_filter.run(
        #     storage = self.storage.step(),
        # )
        
        self.qa_filter.run(
            storage = self.storage.step(),
            input_question = "question",
            input_answer = "answer",
            output_key = "qa_judgement"
        )
        self.qa_filter_processor.run(
            storage = self.storage.step(),
        )
        self.qa_filter_executor.run(
            storage = self.storage.step(),
        )
        
        self.answer_generator.run(
            storage = self.storage.step(),
            input_key = "question", 
            output_key = "llm_answer"
        )
        
        # TODO:
        # 这个judge很有问题，很不准确，得改，可以考虑sympy?
        self.answer_groundtruth_filter.run(
            storage=self.storage.step(), 
            input_test_answer_key="llm_answer",
            input_gt_answer_key="answer",
            input_question_key="question",
          )


def split_generated_content(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 中 'split_qa' 列的 JSON 数组拆分为多行。
    保留原始其他列，并展开每个 sub_question/sub_answer。
    如果 sub_question 或 sub_answer 为空，则不保留该行。
    """
    rows = []
    for _, row in df.iterrows():
        content = row.get("split_qa", None)

        if pd.isna(content) or not str(content).strip():
            continue

        try:
            # 解析 JSON 数组
            items = json.loads(content)
            if not isinstance(items, list):
                items = [items]
        except json.JSONDecodeError:
            print(f"⚠️ JSON parse error in row: {content[:80]}...")
            continue

        for item in items:
            sub_question = item.get("sub_question", "").strip()
            sub_answer = item.get("sub_answer", "").strip()
            
            # 只保留同时存在 sub_question 和 sub_answer 的行
            if not sub_question or not sub_answer:
                continue

            new_row = row.to_dict()
            new_row["sub_id"] = item.get("sub_id", None)
            new_row["sub_question"] = sub_question
            new_row["sub_answer"] = sub_answer
            rows.append(new_row)

    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["sub_id", "sub_question", "sub_answer"])
    
    return pd.DataFrame(rows, columns=list(df.columns) + ["sub_id", "sub_question", "sub_answer"])

def extract_type_and_reason(df: pd.DataFrame) -> pd.DataFrame:
    df["type"] = None
    df["type_reason"] = None

    for idx, row in df.iterrows():
        val = row.get("question_type", "")
        if pd.isna(val) or not str(val).strip():
            continue
        try:
            # 尝试解析 JSON
            j = json.loads(val)
            df.at[idx, "type"] = j.get("type", None)
            df.at[idx, "type_reason"] = j.get("reason", None)
        except json.JSONDecodeError:
            # 如果不是 JSON 格式，尝试按 ":" 分割
            if ":" in val:
                parts = val.split(":", 1)
                df.at[idx, "type"] = parts[0].strip()
                df.at[idx, "type_reason"] = parts[1].strip()
            else:
                df.at[idx, "type"] = val.strip()
                df.at[idx, "type_reason"] = ""

    return df
    
def extract_filter_result_and_reason(df: pd.DataFrame) -> pd.DataFrame:
    df["filter_result"] = None
    df["filter_reason"] = None

    for idx, row in df.iterrows():
        val = row.get("qa_judgement", "")
        if pd.isna(val) or not str(val).strip():
            continue
        try:
            # 尝试解析 JSON
            j = json.loads(val)
            judgement = j.get("judgement", "")
            if isinstance(judgement, bool):
                judgement = "true" if judgement else "false"
            df.at[idx, "filter_result"] = judgement.lower()
            df.at[idx, "filter_reason"] = j.get("reason", None)
        except json.JSONDecodeError:
            df.at[idx, "filter_result"] = ""
            df.at[idx, "filter_reason"] = ""

    return df

if __name__ == "__main__":
    model = BenchSamplingPipeline()
    model.forward()
