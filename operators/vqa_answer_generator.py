from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

from dataflow.prompts.reasoning.math import MathAnswerGeneratorPrompt
from dataflow.prompts.reasoning.general import GeneralAnswerGeneratorPrompt
from dataflow.prompts.reasoning.diy import DiyAnswerGeneratorPrompt
from prompts.question_refine import CaptionPrompt
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

import pandas as pd
from typing import Union, List, Tuple
import re

import os

@prompt_restrict(
    MathAnswerGeneratorPrompt,
    GeneralAnswerGeneratorPrompt,
    CaptionPrompt,
    DiyAnswerGeneratorPrompt
)
@OPERATOR_REGISTRY.register()
class VQAReasoningAnswerGenerator(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self,
                llm_serving: LLMServingABC,
                prompt_template: Union[MathAnswerGeneratorPrompt, GeneralAnswerGeneratorPrompt, DiyAnswerGeneratorPrompt, DIYPromptABC] = MathAnswerGeneratorPrompt,
                skip_text_only: bool=False,
                input_image_default_basedir = "./"
                ):
        
        self.logger = get_logger()
        
        if prompt_template is None:
            prompt_template = MathAnswerGeneratorPrompt()
        self.prompts = prompt_template
        self.llm_serving = llm_serving
        self.skip_text_only = skip_text_only
        self.input_image_default_basedir = input_image_default_basedir
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于为给定问题生成答案，调用大语言模型进行推理。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务实例，用于生成答案\n"
                "- prompt_template：提示模板对象，用于构建生成提示词\n"
                "输出参数：\n"
                "- output_key：生成的答案字段，默认'generated_cot'"
            )
        elif lang == "en":
            return (
                "This operator generates answers for given questions using LLMs for reasoning. \n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving instance for answer generation\n"
                "- prompt_template: Prompt template object for constructing generation prompts\n"
                "Output Parameters:\n"
                "- output_key: Generated answer field, default 'generated_cot'"
            )
        else:
            return "AnswerGenerator produces answers for questions using large language models."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        missing = [k for k in required_keys if k not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")

    def _prepare_vlm_inputs(self, dataframe) -> Tuple[List[str], List[List[str]], List[List[str]], List[int], List[int]]:
        """
        Parses prompts for image markdown, extracts paths and text segments, 
        and structures them into interleaved lists for the VLM server.
        
        返回:
            user_prompts: List[str] (所有的问题)
            list_of_image_paths: List[List[str]] (所有请求的绝对路径列表)
            list_of_text_segments: List[List[str]] (所有图像标签)
            vqa_ids: List[int] (含有图片的问题编号)
            unskipped_ids: List[int] (未跳过的问题编号，这里跳过是指保留已经回答的答案，不重新回答）
        """
        list_of_image_paths: List[List[str]] = []
        list_of_text_segments: List[List[str]] = []
        user_prompts:List[str] = []
        vqa_ids = []
        
        # Markdown 图片正则匹配: ![label](path)
        markdown_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
        
        questions = dataframe[self.input_key].tolist()
        
        unskipped_ids = []
        
        for index, question in enumerate(questions):
            
            # 1. 确定 Base Directory (图像的根目录)
            base_dir = self.input_image_default_basedir
            if self.input_image_basedir_key in dataframe.columns:
                row_base_dir = dataframe.loc[index, self.input_image_basedir_key]
                if row_base_dir:
                    base_dir = row_base_dir
            
            # 2. 准备该请求的结构
            current_paths: List[str] = []
            current_segments: List[str] = []
            current_user_prompt: str = ""
            
            last_end = 0
            
            # 查找所有图片匹配项
            matches = list(markdown_pattern.finditer(question))
            
            # 3. 处理纯文本或交错文本/图像
            if (not matches):
                if (not self.skip_text_only):
                    # 纯文本提示：直接构建提示并作为唯一的文本片段
                    if self.input_skip_key != None and self.input_skip_key in dataframe.columns:
                        if dataframe.loc[index, self.input_skip_key]:
                            continue
                    final_prompt_text = self.prompts.build_prompt(question)
                    user_prompts.append(final_prompt_text)
                    list_of_image_paths.append([])
                    list_of_text_segments.append([])
                    unskipped_ids.append(index)
                continue
            
            vqa_complete = True
            # 4. 遍历匹配项，提取交错的文本片段和图像路径
            for match in matches:
                leading_text = question[last_end:match.start()].strip()
                if leading_text:
                    current_user_prompt += leading_text
                
                label = match.group(1).strip()
                path = match.group(2).strip()
                
                current_segments.append(label)
                
                # 4c. 记录绝对路径 (原始逻辑)
                full_path = os.path.join(base_dir, path)
                
                # =========================================================================
                # [修改开始] 投机取巧的路径修复逻辑 (Heuristic Path Rescue)
                # =========================================================================
                if not os.path.isfile(full_path):
                    # 1. 定义你要求的强制正确根目录 (Hardcoded Correct Root)
                    # 注意：根据你的要求，这里使用了 /jizhicfs/...
                    FORCE_ROOT = "/jizhicfs/herunming/vqa_wzh/images"
                    
                    # 2. 尝试解析路径结构
                    # 你的错误路径包含: .../subset_name/question_images/filename.jpg
                    # 我们尝试提取最后三级：subset_name, question_images, filename
                    
                    try:
                        # 获取文件名 (e.g., xxx.jpg)
                        filename = os.path.basename(full_path)
                        
                        # 获取父目录 (期望是 question_images)
                        parent_dir_path = os.path.dirname(full_path)
                        parent_dir_name = os.path.basename(parent_dir_path)
                        
                        # 获取祖父目录 (期望是 probability_theory_4 这种子集名)
                        grandparent_dir_path = os.path.dirname(parent_dir_path)
                        grandparent_dir_name = os.path.basename(grandparent_dir_path)

                        # 简单的启发式判断：如果路径里包含 question_images，我们就尝试重组
                        if "question_images" in full_path:
                            # 如果当前解析出来的父目录不是 question_images，说明可能路径错位了
                            # 我们尝试在整个字符串里找 question_images 的位置
                            if parent_dir_name != "question_images":
                                # 备用方案：直接字符串分割
                                # 假设路径以 /question_images/filename.jpg 结尾
                                if "/question_images/" in full_path:
                                    parts = full_path.split("/question_images/")
                                    # 取最后一部分作为文件名
                                    filename = parts[-1]
                                    # 取前一部分的最后一个文件夹名作为 subset_name
                                    # parts[0] 可能是 .../probability_theory_4
                                    subset_name = os.path.basename(parts[0])
                                    
                                    # 重组路径
                                    rescue_path = os.path.join(FORCE_ROOT, subset_name, "question_images", filename)
                                else:
                                    rescue_path = None
                            else:
                                # 结构看起来正常，直接用提取出的名字重组
                                rescue_path = os.path.join(FORCE_ROOT, grandparent_dir_name, "question_images", filename)
                            
                            # 3. 检查重组后的路径是否存在
                            if rescue_path and os.path.isfile(rescue_path):
                                self.logger.warning(f"Path Rescue Success: Redirected\nFrom: {full_path}\nTo:   {rescue_path}")
                                full_path = rescue_path
                    except Exception as e:
                        # 如果解析过程出错，不做处理，让它继续走下面的报错流程
                        pass
                # =========================================================================
                # [修改结束]
                # =========================================================================

                # 检查路径是否存在
                if not os.path.isfile(full_path):
                    self.logger.warning(f"Image file not found: {full_path} (from question index {index})")
                    vqa_complete = False
                    break
                
                current_paths.append(full_path)
                
                last_end = match.end()

            trailing_text = question[last_end:].strip()
            if trailing_text:
                current_user_prompt += trailing_text
                
            # 如果caption key存在，添加caption信息
            if self.input_caption_key != None and self.input_caption_key in dataframe.columns:
                captions = dataframe.loc[index, self.input_caption_key]
                for cap_i, caption in enumerate(captions):
                    current_user_prompt += f"\n Description of image {cap_i+1}: {caption}"
                
            # 5. 存储该请求的结果
            if vqa_complete:
                vqa_ids.append(index)
                if self.input_skip_key != None and self.input_skip_key in dataframe.columns:
                    if dataframe.loc[index, self.input_skip_key]:
                        continue
                list_of_image_paths.append(current_paths)
                list_of_text_segments.append(current_segments)
                user_prompts.append(self.prompts.build_prompt(current_user_prompt))
                unskipped_ids.append(index)
                

        return user_prompts, list_of_image_paths, list_of_text_segments, vqa_ids, unskipped_ids

    def run(
        self, 
        storage,
        input_key:str = "instruction", 
        output_key:str = "generated_cot",
        input_caption_key: str | None = None,
        input_skip_key: str | None = None,
        input_image_basedir_key = "image_basedir",
        ):
        '''
        Runs the answer generation process, reading from the input file and saving results to output.
        '''
        self.input_key, self.output_key = input_key, output_key
        self.input_caption_key = input_caption_key
        self.input_skip_key = input_skip_key
        self.input_image_basedir_key = input_image_basedir_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        
        # 1. 准备 VLM 输入: 解析 Markdown 并获取路径和文本片段
        user_prompts, list_of_image_paths, list_of_image_labels, vqa_ids, unskipped_ids = self._prepare_vlm_inputs(dataframe)
        
        # 2. 获取 System Prompt (假设它存储在 self.prompts 对象中)
        # 如果 self.prompts 没有 system_prompt 属性，则使用默认值。
        system_prompt = "You are an intelligent chatbot good at college subjects."
        
        answers = self.llm_serving.generate_from_input_multi_images(
            list_of_image_paths=list_of_image_paths,
            list_of_image_labels=list_of_image_labels, 
            system_prompt=system_prompt,
            user_prompts=user_prompts
        )

        if self.skip_text_only:
            # 只写入vqa_ids指明的行
            dataframe = dataframe.loc[vqa_ids].copy() # 注意，这里不重置索引
        
        # dataframe[self.output_key] = answers
        for idx, ans in zip(unskipped_ids, answers):
            dataframe.at[idx, self.output_key] = ans
                           
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]
