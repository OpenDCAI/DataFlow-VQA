import os
import sys
import json
import json5
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataflow.operators.core_text import PandasOperator, FormatStrPromptedGenerator
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
from prompts.curate_data import TypeClassifyPrompt, SubQuestionSplitingPrompt, QAFilterPrompt
from prompts.question_refine import AddMissingBlankPrompt
from prompts.question_answer_clean import TextCleaningPrompt
from dataflow.operators.core_text import GeneralFilter
import argparse
import re
import shutil


class DataCurationPipeline(PipelineABC):
    def __init__(self, input_file, api_url, model_name, max_workers=100):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path="./cache",
            file_name_prefix="curate_data",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
                api_url=f"{api_url}/chat/completions",
                model_name=model_name,
                max_workers=max_workers,
        )

        self.sub_qa_justify = FormatStrPromptedGenerator(
            llm_serving = self.llm_serving,
            prompt_template = SubQuestionSplitingPrompt()
        )
        self.sub_qa_spliter = PandasOperator(
            [split_generated_content]
        )
        
        # Extract concise answers from solutions
        self.answer_extractor = AnswerExtractionOperator(
            llm_serving=self.llm_serving,
            overwrite=False
        )
        
        # Classify question types
        self.type_filter = FormatStrPromptedGenerator(
            llm_serving = self.llm_serving,
            prompt_template = TypeClassifyPrompt()
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
        
        # Filter items with unverifiable or poorly paired QA
        self.qa_filter = FormatStrPromptedGenerator(
            llm_serving = self.llm_serving,
            prompt_template = QAFilterPrompt()
        )
        self.qa_filter_processor = PandasOperator(
            [extract_filter_result_and_reason]
        )
        self.qa_filter_executor = GeneralFilter(
            filter_rules=[lambda df: df['filter_result'] == 'true']
        )

        # question和answer的非内容型过滤
        self.text_cleaner = LLMTextCleanerOperator(
            llm_serving=self.llm_serving,
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
            input_question_key= "question",
            input_solution_key = "solution",
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
    parser = argparse.ArgumentParser(description="Data Curation Pipeline")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file (raw_vqa.jsonl)")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1", help="Base URL of the OpenAI-compatible API (e.g. https://api.openai.com/v1)")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="LLM model name to use for curation")
    parser.add_argument("--max_workers", type=int, default=100, help="Number of parallel API workers")
    args = parser.parse_args()

    model = DataCurationPipeline(args.input_file, api_url=args.api_url, model_name=args.model, max_workers=args.max_workers)
    model.compile()
    model.forward()
    
    # Find the latest curate_data cache step file
    cache_files = os.listdir("./cache")
    step_files = [f for f in cache_files if re.match(r"curate_data_step\d+\.jsonl", f)]
    step_numbers = [int(re.findall(r"curate_data_step(\d+)\.jsonl", f)[0]) for f in step_files]
    max_step = max(step_numbers)
    max_step_file = f"./cache/curate_data_step{max_step}.jsonl"
    
    # Copy final step file to output directory as curated_vqa.jsonl
    # Output is placed alongside input_file so relative image paths remain valid
    output_dir = os.path.dirname(args.input_file)
    output_file = os.path.join(output_dir, "curated_vqa.jsonl")
    shutil.copy(max_step_file, output_file)
    print(f"Curated data saved to: {output_file}")
