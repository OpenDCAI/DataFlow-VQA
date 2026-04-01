from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from typing import Union

@OPERATOR_REGISTRY.register()
class AnswerExtractionOperator(OperatorABC):
    def __init__(self, llm_serving: Union[None, object] = None, overwrite: bool = False):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.overwrite = overwrite
        self.system_prompt = "You are a professional question answering system. You will be given a question with corresponding solution. Extract a concise and accurate answer from the provided solution. Output only the answer without any additional text."

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于从解答中提取答案，读取解答字段并调用LLM提取答案。"
                "输入参数：\n"
                "- input_solution_key：解答字段名，默认为'solution'\n"
                "- output_key：答案字段名，默认为'answer'\n"
                "- overwrite：是否覆盖已有答案，默认为False\n"
                "输出参数：\n"
                "- output_key：提取的答案"
            )
        elif lang == "en":
            return (
                "This operator extracts answers from solutions, reading from the solution field and using LLM to extract answers."
                "Input Parameters:\n"
                "- input_solution_key: Solution field name, default 'solution'\n"
                "- output_key: Answer field name, default 'answer'\n"
                "- overwrite: Whether to overwrite existing answers, default False\n"
                "Output Parameters:\n"
                "- output_key: Extracted answer"
            )
        else:
            return "AnswerExtractionOperator extracts answers from solutions using LLM."

    def run(self, storage: DataFlowStorage, input_question_key: str = "question", input_solution_key: str = "solution", output_key: str = "answer"):
        dataframe = storage.read("dataframe")

        if input_solution_key not in dataframe.columns:
            raise ValueError(f"input_solution_key: {input_solution_key} not found in dataframe columns.")

        # -----------------------------------
        # 一套统一的空值判断逻辑
        # -----------------------------------
        def _is_valid(x, *, empty_ok=False):
            """
            empty_ok=False: 用来判断 solution 是否有效（空白→无效）
            empty_ok=True: 用来判断 output 是否“空”（空白→空）
            """
            if x is None:
                return empty_ok
            if isinstance(x, float) and pd.isna(x):
                return empty_ok
            if isinstance(x, str) and x.strip() == "":
                return empty_ok
            return not empty_ok

        # solution 必须有效（非空、非空白）
        mask = dataframe[input_solution_key].apply(lambda x: _is_valid(x, empty_ok=False))

        # 若 overwrite=False，则 output_key 为空（空白也算）才处理
        if not self.overwrite and output_key in dataframe.columns:
            mask = mask & dataframe[output_key].apply(lambda x: _is_valid(x, empty_ok=True))

        # 收集有效 solutions
        solutions = dataframe.loc[mask, input_solution_key].tolist()
        questions = dataframe.loc[mask, input_question_key].tolist()

        # 调用 LLM
        if self.llm_serving:
            prompts = [
                self.system_prompt + f"\n\nQuestion: {q}\nSolution: {s}\nNow extract the answer."
                for q, s in zip(questions, solutions)
            ]
            answers = self.llm_serving.generate_from_input(prompts)
        else:
            answers = solutions

        # 写回
        dataframe.loc[mask, output_key] = answers

        output_file = storage.write(dataframe)
        self.logger.info(f"Extracted answers saved to {output_file}")

        return [output_key]