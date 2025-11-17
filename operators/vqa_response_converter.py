import json
import os
import re
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from pathlib import Path
import shutil

@OPERATOR_REGISTRY.register()
class VQAResponseConverter(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
    
    def _id_to_text(self, input_ids, input_json, image_prefix="images"):
        texts = []
        id_list = input_ids.replace(' ', '').split(',')
        for id in id_list:
            try: 
                int(id)
            except:
                continue
            if int(id) < len(input_json):
                try:
                    item = input_json[int(id)]
                except:
                    continue
                if 'text' in item:
                    texts.append(item['text'])
                elif 'img_path' in item:
                    try:
                        img_path = item.get('img_path', '')
                        img_name = os.path.basename(img_path)
                        new_path = f"{image_prefix}/{img_name}"
                        texts.append(f"![{' '.join(item.get('image_caption','image'))}]({new_path})")
                    except:
                        pass
                elif item.get('type','') == 'list':
                    if item['sub_type'] == 'text':
                        try:
                            texts.append(input_json[int(id)]['list_items'].pop(0))
                        except:
                            pass
        return '\n'.join(texts)
    
    def _convert_response(self, input_response, input_json_path, image_prefix="images"):
        qa_list = []
        with open(input_json_path, 'r') as infile:
            input_json = list(json.load(infile))
        # 提取title
        for chapter_block in re.findall(r'<chapter>(.*?)</chapter>', input_response, flags=re.DOTALL):
            title = re.search(r'<title>(.*?)</title>', chapter_block, flags=re.DOTALL)
            if title:
                chapter_title = self._id_to_text(title.group(1).strip(), input_json, image_prefix)
            else:
                chapter_title = ""
            # 找出所有 qa_pair 块
            for pair in re.findall(r'<qa_pair>(.*?)</qa_pair>', chapter_block, flags=re.DOTALL):
                # 提取 question 部分
                q_match = re.search(r'<question>(.*?)</question>', pair, flags=re.DOTALL)
                # 提取 answer 部分
                a_match = re.search(r'<answer>(.*?)</answer>', pair, flags=re.DOTALL)
                # 提取solution部分
                s_match = re.search(r'<solution>(.*?)</solution>', pair, flags=re.DOTALL)
                # 提取label
                label_match = re.search(r'<label>(.*?)</label>', pair, flags=re.DOTALL)
                if not ((q_match and label_match) or (a_match and label_match) or (s_match and label_match)):
                    continue
                label = label_match.group(1).strip()
                qa_list.append({
                    'question': self._id_to_text(q_match.group(1).strip(), input_json, image_prefix) if q_match else "",
                    'answer': a_match.group(1).strip() if a_match else "",
                    'solution': self._id_to_text(s_match.group(1).strip(), input_json, image_prefix) if s_match else "",
                    'label': label,
                    'chapter_title': chapter_title
                })
        return qa_list
    
    def run(self, storage: DataFlowStorage, input_response_key: str = "extracted_qa",
            input_json_path_key: str = "json_path", output_dir_key: str = "output_dir",
            mode_key: str = "mode", output_qa_list_key: str = "qa_list") -> list:
        dataframe = storage.read("dataframe")
        
        if input_response_key not in dataframe.columns:
            raise ValueError(f"Column '{input_response_key}' not found in dataframe")
        if input_json_path_key not in dataframe.columns:
            raise ValueError(f"Column '{input_json_path_key}' not found in dataframe")
        
        responses = dataframe[input_response_key].tolist()
        json_paths = dataframe[input_json_path_key].tolist()
        output_dirs = dataframe[output_dir_key].tolist() if output_dir_key in dataframe.columns else [None] * len(dataframe)
        modes = dataframe[mode_key].tolist() if mode_key in dataframe.columns else ["question"] * len(dataframe)
        
        qa_lists = []
        
        for idx, (response, json_path) in enumerate(zip(responses, json_paths)):
            output_dir = output_dirs[idx] if idx < len(output_dirs) else None
            mode = modes[idx] if idx < len(modes) else "question"
            
            # 确定 image_prefix
            image_prefix = f"{mode}_images"
            
            # 转换 response
            converted_json_path = json_path.replace('.json', '_converted.json')
            qa_list = self._convert_response(response, converted_json_path, image_prefix)
            
            # 复制图片
            if output_dir:
                output_root = output_dir
                if mode in output_dir:
                    output_root = os.path.dirname(output_dir)
                else:
                    output_root = output_dir
                
                src_dir = os.path.join(output_dir, Path(json_path).stem).replace('_content_list','')
                src_images = os.path.join(src_dir, 'vlm', 'images')
                dst_images = os.path.join(output_root, image_prefix)
                
                try:
                    if os.path.exists(src_images):
                        if os.path.exists(dst_images):
                            shutil.rmtree(dst_images)
                        shutil.copytree(src_images, dst_images)
                    else:
                        self.logger.warning(f"Source images dir does not exist: {src_images}")
                except Exception as e:
                    self.logger.warning(f"Failed to copy images from {src_images} to {dst_images}: {e}")
            
            qa_lists.append(qa_list)
        
        dataframe[output_qa_list_key] = qa_lists
        output_file = storage.write(dataframe)
        self.logger.info(f"Response conversion results saved to {output_file}")
        
        return [output_qa_list_key,]

