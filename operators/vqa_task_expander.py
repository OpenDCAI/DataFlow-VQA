import os
import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class VQATaskExpander(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
    
    def run(self, storage: DataFlowStorage, 
            question_pdf_path_key: str = "question_pdf_path",
            answer_pdf_path_key: str = "answer_pdf_path",
            pdf_path_key: str = "pdf_path",  # 支持 interleaved 模式的单一 pdf_path
            subject_key: str = "subject",
            output_dir_key: str = "output_dir",
            output_pdf_path_key: str = "pdf_path",
            output_mode_key: str = "mode",
            output_interleaved_key: str = "interleaved") -> list:
        dataframe = storage.read("dataframe")
        
        # 支持两种输入格式：question_pdf_path/answer_pdf_path 或 pdf_path
        if question_pdf_path_key not in dataframe.columns and pdf_path_key not in dataframe.columns:
            raise ValueError(f"Column '{question_pdf_path_key}' or '{pdf_path_key}' not found in dataframe")
        
        expanded_rows = []
        
        for idx, row in dataframe.iterrows():
            # 优先使用 question_pdf_path，如果没有则使用 pdf_path（interleaved 模式）
            if question_pdf_path_key in dataframe.columns:
                question_pdf_path = row[question_pdf_path_key]
                answer_pdf_path = row.get(answer_pdf_path_key, question_pdf_path)
            else:
                # interleaved 模式：使用同一个 pdf_path
                question_pdf_path = row[pdf_path_key]
                answer_pdf_path = question_pdf_path
            
            subject = row.get(subject_key, "General")
            output_root = row.get(output_dir_key, "../vqa_output")
            interleaved = (question_pdf_path == answer_pdf_path)
            
            os.makedirs(output_root, exist_ok=True)
            
            # Question task
            q_outdir = os.path.join(output_root, "question")
            os.makedirs(q_outdir, exist_ok=True)
            expanded_rows.append({
                output_pdf_path_key: question_pdf_path,
                output_mode_key: "question",
                output_interleaved_key: interleaved,
                subject_key: subject,
                output_dir_key: q_outdir,
                "output_root": output_root
            })
            
            # Answer task (if not interleaved)
            if not interleaved:
                a_outdir = os.path.join(output_root, "answer")
                os.makedirs(a_outdir, exist_ok=True)
                expanded_rows.append({
                    output_pdf_path_key: answer_pdf_path,
                    output_mode_key: "answer",
                    output_interleaved_key: interleaved,
                    subject_key: subject,
                    output_dir_key: a_outdir,
                    "output_root": output_root
                })
        
        expanded_dataframe = pd.DataFrame(expanded_rows)
        output_file = storage.write(expanded_dataframe)
        self.logger.info(f"Expanded {len(dataframe)} rows to {len(expanded_dataframe)} tasks. Saved to {output_file}")
        
        return [output_pdf_path_key, output_mode_key, output_interleaved_key, subject_key, output_dir_key]

