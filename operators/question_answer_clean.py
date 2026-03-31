import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import re


@OPERATOR_REGISTRY.register()
class LLMTextCleanerOperator(OperatorABC):
    def __init__(
            self,
            llm_serving: LLMServingABC,
            prompt_template,
            max_batch_size: int = 32
        ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template
        self.max_batch_size = max_batch_size
        
        if prompt_template is None:
            raise ValueError("prompt_template cannot be None")

    def apply_deletions(self, original_text, deletion_output):
        """从原始文本中删除指定片段"""
        if not deletion_output or deletion_output.strip() == "NONE":
            return original_text
        # 按 || 分割片段
        fragments = [frag.strip() for frag in deletion_output.split("||") if frag.strip()]
        # 按长度降序排序，避免短串误删长串的一部分
        fragments = sorted(fragments, key=len, reverse=True)
        cleaned = original_text
        for frag in fragments:
            cleaned = cleaned.replace(frag, "", 1)  # 只删除一次
        return cleaned

    def run(
            self, 
            storage: DataFlowStorage,
            output_key: str = "cleaned_dataframe",
            question_column: str = "question",
            answer_column: str = "answer",
            **input_keys
        ):
        self.storage: DataFlowStorage = storage
        self.output_key = output_key
        self.question_column = question_column
        self.answer_column = answer_column
        self.logger.info("Running LLMTextCleanerOperator...")

        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading dataframe, number of rows: {len(dataframe)}")

        if len(dataframe) == 0:
            self.logger.warning("No data to process")
            output_file = storage.write(dataframe)
            return output_key

        question_prompts = []
        answer_prompts = []
        valid_indices = []

        for idx, row in dataframe.iterrows():
            question = str(row.get(question_column, ""))
            answer = str(row.get(answer_column, ""))
            
            q_prompt = self.prompt_template.build_question_prompt(question)
            a_prompt = self.prompt_template.build_answer_prompt(answer)
            
            question_prompts.append(q_prompt)
            answer_prompts.append(a_prompt)
            valid_indices.append(idx)

        self.logger.info(f"Prepared {len(question_prompts)} question prompts and {len(answer_prompts)} answer prompts")

        question_deletion_outputs = self.llm_serving.generate_from_input(question_prompts)
        self.logger.info("Completed question cleaning prompts processing")

        answer_deletion_outputs = self.llm_serving.generate_from_input(answer_prompts)
        self.logger.info("Completed answer cleaning prompts processing")

        cleaned_questions = []
        cleaned_answers = []

        for i in range(len(question_deletion_outputs)):
            original_question = str(dataframe.iloc[i][question_column])
            original_answer = str(dataframe.iloc[i][answer_column])
            
            cleaned_q = self.apply_deletions(original_question, question_deletion_outputs[i])
            cleaned_a = self.apply_deletions(original_answer, answer_deletion_outputs[i])
            
            cleaned_questions.append(cleaned_q.strip())
            cleaned_answers.append(cleaned_a.strip())

        result_dataframe = dataframe.copy()
        result_dataframe[question_column] = cleaned_questions
        result_dataframe[answer_column] = cleaned_answers

        output_file = storage.write(result_dataframe)
        self.logger.info(f"Cleaning completed, processed {len(result_dataframe)} rows")

        return output_key