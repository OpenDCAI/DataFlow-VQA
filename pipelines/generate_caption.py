import os
import sys
import json5
import pandas as pd
from transformers import Pipeline
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from operators.vqa_answer_generator import VQAReasoningAnswerGenerator


from dataflow.pipeline import PipelineABC   
from dataflow.serving import APILLMServing_request, APIVLMServing_openai
from dataflow.operators.core_text import PandasOperator
from dataflow.utils.storage import FileStorage
from dataflow import get_logger

from prompts.question_refine import CaptionPrompt

import re

def make_get_caption_and_remove_irrelevant_image(question_key="question", caption_key="captions"):
    pattern = r'!\[[^\]]*\]\(([^)]+)\)'
    logger = get_logger()
    def fn(df):
        df = df.copy()
        for idx, row in df.iterrows():
            question = row[question_key]
            captions = row[caption_key]
            try:
                captions = json5.loads(captions)
                assert isinstance(captions, list)
            except:
                logger.error(f"Error parsing captions for row {idx}")
                df.at[idx, caption_key] = []
                continue
                
            matches = list(re.finditer(pattern, question))
            
            if len(matches) != len(captions):
                logger.warning(f"Row {idx}: {len(matches)} image patterns vs {len(captions)} captions")

            # Remove matched image markdowns whose corresponding caption is "IRRELEVANT".
            # Iterate in reverse so removals don't affect earlier match indices.
            count = min(len(matches), len(captions))
            for j in range(count - 1, -1, -1):
                m = matches[j]
                cap = captions[j]
                if isinstance(cap, str) and cap.strip().upper() == "IRRELEVANT":
                    logger.info(f"Row {idx}: removing IRRELEVANT image '{m.group(0)}'")
                    question = question[:m.start()] + question[m.end():]

            # cleanup extra whitespace left after removals
            question = re.sub(r'\s{2,}', ' ', question).strip()
            df.at[idx, question_key] = question
            df.at[idx, caption_key] = captions

        return df
    
    return fn

class CaptionGeneratingPipeline(PipelineABC):
    def __init__(self, first_entry_file_name, cache_path, file_name_prefix, input_image_default_basedir, cache_type="json"):
        super().__init__()
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
                timeout=120,
                temperature=1.0
        )
        
        self.input_image_default_basedir = input_image_default_basedir
        
        
        #llm 产生caption
        self.caption_generator = VQAReasoningAnswerGenerator(
            llm_serving=self.llm_answer_serving,
            prompt_template=CaptionPrompt(),
            skip_text_only=True,
            input_image_default_basedir=self.input_image_default_basedir,
        )
        
        self.post_processor = PandasOperator(process_fn=[ make_get_caption_and_remove_irrelevant_image(question_key="question", caption_key="captions") ])
        
    def forward(self):
                
        self.caption_generator.run(
            storage = self.storage.step(),
            input_key = "question", 
            output_key = "captions",
        )
        
        self.post_processor.run(
            storage = self.storage.step(),
        )

if __name__ == "__main__":
    first_entry_file_name=f"/data1/djw/VQA_1209/vision_only_shuffled_top100.json"
    cache_path = f"/data1/djw/VQA_1209/caption_cache"
    file_name_prefix = f"gpt-5-mini"
    input_image_default_basedir = f"./"
    
    model = CaptionGeneratingPipeline(first_entry_file_name, cache_path, file_name_prefix, input_image_default_basedir)
    model.compile()
    model._compiled_forward(resume_step=1)