import os
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from operators.vqa_extract_doclayout import VQAExtractDocLayoutMinerU
from typing import Literal

@OPERATOR_REGISTRY.register()
class VQALayoutExtractor(OperatorABC):
    def __init__(self, mineru_backend: Literal["vlm-transformers","vlm-vllm-engine"] = "vlm-transformers"):
        self.logger = get_logger()
        self.mineru_backend = mineru_backend
        self.doc_layout = VQAExtractDocLayoutMinerU(mineru_backend=mineru_backend)
    
    def run(self, storage: DataFlowStorage, input_pdf_path_key: str = "pdf_path", 
            output_dir_key: str = "output_dir", output_json_path_key: str = "json_path",
            mode_key: str = "mode") -> list:
        dataframe = storage.read("dataframe")
        
        if input_pdf_path_key not in dataframe.columns:
            raise ValueError(f"Column '{input_pdf_path_key}' not found in dataframe")
        
        pdf_paths = dataframe[input_pdf_path_key].tolist()
        output_dirs = dataframe[output_dir_key].tolist() if output_dir_key in dataframe.columns else [None] * len(dataframe)
        modes = dataframe[mode_key].tolist() if mode_key in dataframe.columns else ["question"] * len(dataframe)
        
        json_paths = []
        
        for idx, pdf_path in enumerate(pdf_paths):
            output_dir = output_dirs[idx] if idx < len(output_dirs) else None
            mode = modes[idx] if idx < len(modes) else "question"
            
            if output_dir is None:
                # 默认输出目录
                output_dir = os.path.join(os.path.dirname(pdf_path), mode)
            
            os.makedirs(output_dir, exist_ok=True)
            
            json_path, layout_path = self.doc_layout.run(
                storage=None,
                input_pdf_file_path=pdf_path,
                output_folder=output_dir
            )
            
            json_paths.append(json_path)
        
        dataframe[output_json_path_key] = json_paths
        output_file = storage.write(dataframe)
        self.logger.info(f"Layout extraction results saved to {output_file}")
        
        return [output_json_path_key,]

