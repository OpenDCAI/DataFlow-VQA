import os
import sys
import json
import json5
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataflow.operators.core_text import PandasOperator, PromptTemplatedGenerator
from operators.bench_evaluate import BenchDatasetEvaluatorQuestion
from operators.answer_extractor import AnswerExtractionOperator
from operators.question_refiner import AddMissingBlankOperator
from operators.question_answer_clean import LLMTextCleanerOperator

from dataflow.pipeline import PipelineABC
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.reasoning import (
    ReasoningAnswerGenerator,
    ReasoningAnswerGroundTruthFilter
)
from dataflow.prompts.reasoning.general import GeneralAnswerGeneratorPrompt
from prompts.bench_sampling import BenchSamplingPrompt, SubQuestionSplitingPrompt, QAFilterPrompt
from prompts.question_refine import AddMissingBlankPrompt
from prompts.question_answer_clean import TextCleaningPrompt
from dataflow.prompts.core_text import StrFormatPrompt
from dataflow.operators.core_text import GeneralFilter
import argparse

class BenchSamplingPipeline(PipelineABC):
    def __init__(self, first_entry_file_name, cache_path, file_name_prefix, cache_type="json"):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name, ### 现在dataflow还不支持图架构，难以把qa，qas，qs的分类放进来，这个算法在 /data1/hzh/vqa/completeness_filter.py
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        self.llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-5-mini-2025-08-07",
                max_workers=100,
        )
        
        self.llm_answer_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-5-mini",
                max_workers=100,
        )
        self.llm_clean_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="deepseek-v3.2",
                max_workers=100,
        )

        # 拆小题
        self.sub_qa_justify = PromptTemplatedGenerator(
            llm_serving = self.llm_serving,
            prompt_template = SubQuestionSplitingPrompt()
        )
        self.sub_qa_spliter = PandasOperator(
            [split_generated_content]
        )
        
        # 抽取答案
        self.answer_extractor = AnswerExtractionOperator(
            llm_serving=self.llm_serving,
            overwrite=False
        )
        
        # 判断题型
        self.type_filter = PromptTemplatedGenerator(
            llm_serving = self.llm_serving,
            prompt_template = BenchSamplingPrompt()
        )
        self.type_filter_processor = PandasOperator(
            [extract_type_and_reason]
        )
        self.type_filter_executor = GeneralFilter(
            filter_rules=[lambda df: df['type'].isin(["Calculation", "Fill-in", "Multiple-choice"])]
        )
        
        self.add_missing_blank = AddMissingBlankOperator(
            llm_serving=self.llm_serving,
            prompt_template=AddMissingBlankPrompt()
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
            prompt_template=GeneralAnswerGeneratorPrompt()
        )
        
        ## llm核对
        self.answer_groundtruth_filter = BenchDatasetEvaluatorQuestion(
            compare_method="semantic",
            llm_serving=self.llm_serving,
            prompt_template=None, # using default prompt
            eval_result_path="../eval_result_math_test.json",
            support_subquestions=True
        )

        # question和answer的非内容型过滤
        self.text_cleaner = LLMTextCleanerOperator(
            llm_serving=self.llm_serving,           # gpt-5-mini 效果不好可以换成 llm_clean_serving
            prompt_template=TextCleaningPrompt()
        )
        
    def forward(self):
        self.sub_qa_justify.run(
            storage = self.storage.step(),
            output_key = "split_qa",
            input_question  = "question",
            input_answer = "answer",
            input_solution = "solution",
        )
        
        self.sub_qa_spliter.run(
            storage = self.storage.step(),
        )
        
        
        self.type_filter.run(
            storage = self.storage.step(),
            input_question = "question",
            input_answer = "answer",
            output_key = "question_type"
        )       
        self.type_filter_processor.run(
            storage = self.storage.step(),
        )        
        self.type_filter_executor.run(
            storage = self.storage.step(),
        )
        
        self.answer_extractor.run(
            storage = self.storage.step(),
            input_key = "solution",
            output_key= "answer"
        )
        
        self.add_missing_blank.run(
            storage = self.storage.step(),
            input_question = "question",
            input_answer = "answer",
            output_key = "question",
        )
        
        self.text_cleaner.run(
            storage=self.storage.step(),
            question_column="question",
            answer_column="answer",
            output_key="cleaned_dataframe"
        )
        
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
                
        # self.answer_generator.run(
        #     storage = self.storage.step(),
        #     input_key = "question", 
        #     output_key = "llm_answer"
        # )
        
        # # TODO:
        # # 这个judge很有问题，很不准确，得改，可以考虑sympy?
        # self.answer_groundtruth_filter.run(
        #     storage=self.storage.step(), 
        #     input_test_answer_key="llm_answer",
        #     input_gt_answer_key="answer",
        #     input_question_key="question",
        #   )


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
            items = json5.loads(content)
            if not isinstance(items, list):
                items = [items]
        except:
            print(f"⚠️ JSON parse error in row: {content[:80]}...")
            continue

        for item in items:
            sub_question = item.get("sub_question", "").strip()
            sub_answer = item.get("sub_answer", "").strip()
            sub_solution = item.get("sub_solution", "").strip()
            
            # 只保留同时存在 sub_question 和 sub_answer 的行
            if not sub_question or not (sub_answer or sub_solution):
                continue

            new_row = row.to_dict()
            # new_row["sub_id"] = item.get("sub_id", None)
            new_row["question"] = sub_question if sub_question != "ORIGINAL" else row["question"]
            new_row["answer"] = sub_answer if sub_answer != "ORIGINAL" else row["answer"]
            new_row["solution"] = sub_solution if sub_solution != "ORIGINAL" else row.get("solution", "")
            rows.append(new_row)

    if not rows:
        return pd.DataFrame(columns=list(df.columns))
    
    return pd.DataFrame(rows, columns=list(df.columns))

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
    parser = argparse.ArgumentParser(description="Run BenchSamplingPipeline")
    parser.add_argument("name", nargs="?", default="pde", help="dataset name (default: pde)")
    parser.add_argument("end", nargs="?", type=int, default=3, help="range end for i (uses range(1, end + 1), default: 3)")
    args = parser.parse_args()
    
    # Example usage:
    # python bench_sampling.py pde 3

    name = args.name

    # for i in range(1, args.end + 1):
    
    #     first_entry_file_name=f"/data1/VQA_ready_data/{name}_{i}/vqa_filtered_qa_pairs.jsonl"
    #     cache_path = f"/data1/VQA_ready_data/{name}_{i}"
    #     file_name_prefix = f"{name}_{i}"
        
    #     model = BenchSamplingPipeline(first_entry_file_name, cache_path, file_name_prefix)
    #     model.compile()
    #     model.forward(resume_step=10)
    
    ##############################################################################
    # 重要：现在改成了只跑一个数字，而不是学科内的多个数字全跑，增加灵活性
    ##############################################################################
    
    end = args.end
    
    first_entry_file_name=f"/data1/VQA_ready_data/{name}_{end}/vqa_filtered_qa_pairs.jsonl"
    cache_path = f"/data1/VQA_ready_data/{name}_{end}"
    file_name_prefix = f"{name}_{end}"
    
    model = BenchSamplingPipeline(first_entry_file_name, cache_path, file_name_prefix)
    model.compile()
    model.forward(resume_step=7)
