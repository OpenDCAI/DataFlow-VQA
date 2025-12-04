import json
import re
import os

def merge_qa_pair(question_jsonl, answer_jsonl, output_jsonl):
    with open(question_jsonl, 'r', encoding='utf-8') as q_file, open(answer_jsonl, 'r', encoding='utf-8') as a_file, open(output_jsonl, 'w', encoding='utf-8') as out_file:
        chapter_id = 0
        chapter_title = ""
        label = 1000000
        questions = {}
        answers = {}
        for line in q_file:
            data = json.loads(line)
            label_match = re.search(r'\d+', data["label"])
            if label_match:
                data["label"] = label_match.group()
            if data["chapter_title"] == "":
                data["chapter_title"] = chapter_title
            
            try:
                data["label"] = int(data["label"])
            except:
                continue
            
            if data["chapter_title"] != "" and data["chapter_title"] != chapter_title:
                if data["label"] < label:
                    chapter_id += 1
                    chapter_title = data["chapter_title"]
                else:
                    # 如果题号增加，章节标题却发生变化，说明可能错误提取了子标题。因此继续使用之前的章节标题。
                    data["chapter_title"] = chapter_title
            label = data["label"]
            data["chapter_id"] = chapter_id
            # TODO : 这里可能需要更复杂的title清洗逻辑
            # 删除title中的空格与换行符
            data["chapter_title"] = re.sub(r'\s+', '', data["chapter_title"])
            try:
                data["chapter_id"] = re.search(r"\d+\.\d+|\d+", data["chapter_title"]).group()
            except:
                try:
                    data["chapter_id"] = re.search(r'[一二三四五六七八九零十百]+', data["chapter_title"]).group()   
                except:
                    data["chapter_id"] = data["chapter_title"]
            if data['label'] > 0 and data["chapter_id"]:
                # 已经完整的题目直接写入out_file
                if data["answer"] or data["solution"]:
                    qa_pair = {
                        "question_chapter_id": data["chapter_id"],
                        "answer_chapter_id": data["chapter_id"],
                        "label": data['label'],
                        "question": data["question"],
                        "answer": data["answer"],
                        "solution": data.get("solution", "")
                    }
                    out_file.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                    
                else:
                    questions[(data["chapter_id"], data['label'])] = data
        
        chapter_id = 0
        chapter_title = ""
        label = 1000000
        for line in a_file:
            data = json.loads(line)
            label_match = re.search(r'\d+', data["label"])
            if label_match:
                data["label"] = label_match.group()
            if data["chapter_title"] == "":
                data["chapter_title"] = chapter_title
                
            try:
                data["label"] = int(data["label"])
            except:
                continue
            
            if data["chapter_title"] != "" and data["chapter_title"] != chapter_title:
                if data["label"] < label:
                    chapter_id += 1
                    chapter_title = data["chapter_title"]
                else:
                    # 如果题号增加，章节标题却发生变化，说明可能错误提取了子标题。因此继续使用之前的章节标题。
                    data["chapter_title"] = chapter_title
            label = data["label"]
            data["chapter_id"] = chapter_id
            # TODO : 这里可能需要更复杂的title清洗逻辑
            # 删除title中的空格与换行符
            data["chapter_title"] = re.sub(r'\s+', '', data["chapter_title"])
            try:
                # 优先提取阿拉伯数字章节编号（如1.1，2等）
                data["chapter_id"] = re.search(r"\d+\.\d+|\d+", data["chapter_title"]).group()
            except:    
                try:
                    # 其次提取中文数字章节编号（如六、二十四等）
                    data["chapter_id"] = re.search(r'[一二三四五六七八九零十百]+', data["chapter_title"]).group()   
                except:
                    data["chapter_id"] = data["chapter_title"]
            # 动态更新，防止错误的重复label覆盖掉之前的solution或answer
            if data['label'] > 0:
                if not answers.get((data["chapter_id"], data['label'])):
                    answers[(data["chapter_id"], data['label'])] = data
                else:
                    if not answers[(data["chapter_id"], data['label'])].get("solution") and data.get("solution"):
                        answers[(data["chapter_id"], data['label'])]["solution"] = data["solution"]
                    if not answers[(data["chapter_id"], data['label'])].get("answer") and data.get("answer"):
                        answers[(data["chapter_id"], data['label'])]["answer"] = data["answer"]
      
        for label in questions:
            if label in answers:
                qa_pair = {
                    "question_chapter_id": questions[label]["chapter_id"],
                    "answer_chapter_id": answers[label]["chapter_id"],
                    "label": label[1],
                    "question": questions[label]["question"],
                    "answer": answers[label]["answer"],
                    "solution": answers[label].get("solution", "")
                }
                out_file.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
        
        print(f"Merged QA pairs: {len(questions.keys() & answers.keys())}")
        
def jsonl_to_md(jsonl_file, md_file):
    with open(jsonl_file, 'r', encoding='utf-8') as in_file, open(md_file, 'w', encoding='utf-8') as out_file:
        for line in in_file:
            data = json.loads(line)
            out_file.write(f"### Question {data['label']}\n\n")
            out_file.write(f"{data['question']}\n\n")
            out_file.write(f"**Answer:** {data['answer']}\n\n")
            if data.get('solution'):
                out_file.write(f"**Solution:**\n\n{data['solution']}\n\n")
            out_file.write("---\n\n")
            
if __name__ == "__main__":
    output_root = "/data1/VQA_ready_data/operational_research_2/"
    
    question_jsonl = os.path.join(output_root, "vqa_extracted_questions.jsonl")
    answer_jsonl = os.path.join(output_root, "vqa_extracted_answers.jsonl")
    output_jsonl = os.path.join(output_root, "vqa_merged_qa_pairs.jsonl")
    merge_qa_pair(question_jsonl, answer_jsonl, output_jsonl)