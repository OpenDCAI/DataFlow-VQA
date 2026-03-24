## VQA Extraction
在根目录下跑
```bash
python -m pipelines.vqa_extract_optimized_pipeline --input_file ./examples/VQA/vqa_extract_test.jsonl --output_dir ./output
```
请先在代码中配置模型。并使用`export DF_API_KEY=your_api_key`的方式设置好环境变量。
可以extract VQA，注意进入代码中按照要求准备好输入的jsonl文件！
示例：
```
{"input_pdf_paths": "./examples/VQA/questionextract_test.pdf", "name": "math1"}
{"input_pdf_paths": ["./examples/VQA/math_question.pdf", "./examples/VQA/math_answer.pdf"], "name": "math2"}
```
其中`input_pdf_paths`可以是单个pdf文件，也可以是一个pdf文件列表（题目在前答案在后）。`name`是一个标识符。

切割出的VQA会存到`output_dir`中的`raw_vqa.jsonl`中, 图片也保存在`output_dir`中，jsonl中的`image_basedir`项会指向图片的存储base路径。

## Data Curation
```python
python -m pipelines.curate_data --input_file ./output/raw_vqa.jsonl
```
请先在代码中配置模型。并使用`export DF_API_KEY=your_api_key`的方式设置好环境变量。
使用刚才extract出来的`raw_vqa.jsonl`作为输入文件，跑完之后会在同一目录下生成一个`curated_vqa.jsonl`，里面是经过过滤和清洗后的VQA数据，可以直接拿来做rollout了。
这部分目前分为4个部分：[切小题](#切小题)，[判断题型](#判断题型)，[抽取答案](#抽取答案)，[题目过滤](#题目过滤)。

### 切小题
```python
self.sub_qa_justify = PromptTemplatedGenerator(
    llm_serving = self.llm_serving,
    prompt_template = SubQuestionSplitingPrompt()
)
self.sub_qa_spliter = PandasOperator(
    [split_generated_content]
)
```
目前会把前后无*明显*依赖关系的问题切开，并配上相应的答案和解答过程。

解答过程目前存在割裂的情况，是之后做dataset时需要进一步关注的点。

这一步的新增项：`split_qa`

输出示例

```json
{
    "question_chapter_title":"17Exercises",
    "answer_chapter_title":"17Exercises",
    "label":53,
    "question":"53. Define  $T: \\mathbb{R}^2 \\to \\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . What are the fixed points of  $T$ ? In which quadrants of the  $xy$ -plane is  $T$  a contraction?",
    "answer":"$(0,0),(1,1),(-1, - 1)$",
    "solution":"53.  $(0,0),(1,1),(-1, - 1)$",
    "split_qa":"[\n  {\n    \"sub_id\": 1,\n    \"sub_question\": \"Define  $T: \\\\mathbb{R}^2 \\\\to \\\\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . What are the fixed points of  $T$ ?\",\n    \"sub_answer\": \"$(0,0),(1,1),(-1, - 1)$\",\n    \"sub_solution\": \"53.  $(0,0),(1,1),(-1, - 1)$\"\n  },\n  {\n    \"sub_id\": 2,\n    \"sub_question\": \"Define  $T: \\\\mathbb{R}^2 \\\\to \\\\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . In which quadrants of the  $xy$ -plane is  $T$  a contraction?\",\n    \"sub_answer\": \"\",\n    \"sub_solution\": \"\"\n  }\n]"
  },
```

此后，`split_generated_content`会把小题提取出来，并自动过滤掉question为空，或者answer与solution均为空的项。

例如：

```json
{
    "question_chapter_title":"17Exercises",
    "answer_chapter_title":"17Exercises",
    "label":53,
    "question":"Define  $T: \\mathbb{R}^2 \\to \\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . What are the fixed points of  $T$ ?",
    "answer":"$(0,0),(1,1),(-1, - 1)$",
    "solution":"53.  $(0,0),(1,1),(-1, - 1)$",
    "split_qa":"[\n  {\n    \"sub_id\": 1,\n    \"sub_question\": \"Define  $T: \\\\mathbb{R}^2 \\\\to \\\\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . What are the fixed points of  $T$ ?\",\n    \"sub_answer\": \"$(0,0),(1,1),(-1, - 1)$\",\n    \"sub_solution\": \"53.  $(0,0),(1,1),(-1, - 1)$\"\n  },\n  {\n    \"sub_id\": 2,\n    \"sub_question\": \"Define  $T: \\\\mathbb{R}^2 \\\\to \\\\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . In which quadrants of the  $xy$ -plane is  $T$  a contraction?\",\n    \"sub_answer\": \"\",\n    \"sub_solution\": \"\"\n  }\n]"
  },
```

### 判断题型
```python
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
```

这一步会判断题型（Calculation | Proof | Explanation | Fill-in | Multiple-choice | Sketching/Plotting | Other），然后只保留`filter_rules`中规定的种类（可以自行修改）。

这一步新增的项是`type`和`type_reason`.

输出示例
```json
{
    "question_chapter_title":"17Exercises",
    "answer_chapter_title":"17Exercises",
    "label":53,
    "question":"Define  $T: \\mathbb{R}^2 \\to \\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . What are the fixed points of  $T$ ?",
    "answer":"$(0,0),(1,1),(-1, - 1)$",
    "solution":"53.  $(0,0),(1,1),(-1, - 1)$",
    "split_qa":"[\n  {\n    \"sub_id\": 1,\n    \"sub_question\": \"Define  $T: \\\\mathbb{R}^2 \\\\to \\\\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . What are the fixed points of  $T$ ?\",\n    \"sub_answer\": \"$(0,0),(1,1),(-1, - 1)$\",\n    \"sub_solution\": \"53.  $(0,0),(1,1),(-1, - 1)$\"\n  },\n  {\n    \"sub_id\": 2,\n    \"sub_question\": \"Define  $T: \\\\mathbb{R}^2 \\\\to \\\\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . In which quadrants of the  $xy$ -plane is  $T$  a contraction?\",\n    \"sub_answer\": \"\",\n    \"sub_solution\": \"\"\n  }\n]",
    "question_type":"{\n  \"type\": \"Calculation\",\n  \"reason\": \"Finding fixed points requires solving the system (x,y) = (y^{1\/3}, x^{1\/3}), i.e. straightforward algebraic computation to determine the solutions.\"\n}",
    "type":"Calculation",
    "type_reason":"Finding fixed points requires solving the system (x,y) = (y^{1\/3}, x^{1\/3}), i.e. straightforward algebraic computation to determine the solutions."
  },
```

### 抽取答案
```python
self.answer_extractor = AnswerExtractionOperator(
    llm_serving=self.llm_serving,
    overwrite=False
)
```
这一步会把solution中的answer提取出来，直接写入answer项。`overwrite=True`会对answer已经存在的项进行操作，`overwrite=False`则会跳过这些项。

这一步没有新增的项。

### 题目过滤
```python
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
```

这一步会过滤掉不是问题的内容（也包括示例、开放性问题）、问题或答案依赖外部文本、问题和答案看起来不配对的pair。

目前的prompt：
```txt
[Criteria]
1. The question must be suitable for an exam setting, meaning it should raise a clear problem that requires a specific solution.
    Examples, statements without questions, open-ended discussions and other context that do not pose a clear problem are not suitable.
    Questions like "Give an example of..." that can have many valid answers are also not suitable.
2. Relevance: The answer must directly address the question asked.
    If the answer seems to be addressing a different question and is wrongly paired with the given question, it is not suitable.
3. Completeness and Self-Containment: The question and answer should be complete and self-contained, providing all necessary information for understanding and solving it without requiring external context.
   Questions that rely heavily on prior context or external references are not suitable.
   Answers such as "Refer to theorem X", "Corollary of previous result", "Answered in the text above", "Omitted for brevity" are not acceptable.
   Incomplete questions or answers that leave out critical information are also not suitable.
   
[Important Notice]
1. You do not need to evaluate the correctness of the answer, only whether it is appropriate and complete in relation to the question.
2. Short answer with no explanation (calculation, proof, counterexample, ...) is acceptable as long as it directly addresses the question.
```

这一步新增的项是`filter_result`和`filter_reason`.

输出示例
```json
{
    "question_chapter_title":"17Exercises",
    "answer_chapter_title":"17Exercises",
    "label":53,
    "question":"Define  $T: \\mathbb{R}^2 \\to \\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . What are the fixed points of  $T$ ?",
    "answer":"$(0,0),(1,1),(-1, - 1)$",
    "solution":"53.  $(0,0),(1,1),(-1, - 1)$",
    "split_qa":"[\n  {\n    \"sub_id\": 1,\n    \"sub_question\": \"Define  $T: \\\\mathbb{R}^2 \\\\to \\\\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . What are the fixed points of  $T$ ?\",\n    \"sub_answer\": \"$(0,0),(1,1),(-1, - 1)$\",\n    \"sub_solution\": \"53.  $(0,0),(1,1),(-1, - 1)$\"\n  },\n  {\n    \"sub_id\": 2,\n    \"sub_question\": \"Define  $T: \\\\mathbb{R}^2 \\\\to \\\\mathbb{R}^2$  by  $T(x, y) = (y^{1\/3}, x^{1\/3})$ . In which quadrants of the  $xy$ -plane is  $T$  a contraction?\",\n    \"sub_answer\": \"\",\n    \"sub_solution\": \"\"\n  }\n]",
    "question_type":"{\n  \"type\": \"Calculation\",\n  \"reason\": \"Finding fixed points requires solving the system (x,y) = (y^{1\/3}, x^{1\/3}), i.e. straightforward algebraic computation to determine the solutions.\"\n}",
    "type":"Calculation",
    "type_reason":"Finding fixed points requires solving the system (x,y) = (y^{1\/3}, x^{1\/3}), i.e. straightforward algebraic computation to determine the solutions.",
    "qa_judgement":"{\n  \"reason\": \"The question poses a clear, specific problem suitable for an exam (find fixed points of a given map). The provided answer directly addresses that question with a complete list of fixed points and is self-contained.\",\n  \"judgement\": \"true\"\n}",
    "filter_result":"true",
    "filter_reason":"The question poses a clear, specific problem suitable for an exam (find fixed points of a given map). The provided answer directly addresses that question with a complete list of fixed points and is self-contained."
  }
```

## Generate COT
请先在代码中配置模型。并使用`export DF_API_KEY=your_api_key`的方式设置好环境变量。
```python
python -m pipelines.generate_cot --input_file ./output/curated_vqa.jsonl --max_retries 5
```

这部分分为3步：

- llm 回答:
  ```python
  self.answer_generator = VQAReasoningAnswerGenerator(
      llm_serving=self.llm_answer_serving,
      prompt_template=MathAnswerGeneratorPrompt(),
      skip_text_only=True,
  )
  ```
  设置`skip_text_only=True`会跳过非VQA（问题中没有图片），`skip_text_only=False`则会跑所有的题目。

  llm的回答会存入`llm_answer`当中。 
- 清理llm answer中的thinking部分. `self.think_cleaner = PandasOperator(process_fn=[ make_remove_think_fn() ])`

  为了节省verify成本，这个算子可以把模型回答中第一个`</think>`前面的内容删掉。这里假设输出为`THINK</think>ANSWER`，没有前导的`<think>`。这部分逻辑也可以自己修改。

  清理后的回答会存入`llm_short_answer`当中。
- llm verify：
```python
self.answer_groundtruth_filter = BenchDatasetEvaluatorQuestion(
  compare_method="semantic",
  llm_serving=self.llm_serving,
  prompt_template=None, # using default prompt
  eval_result_path=eval_result_path,
  support_subquestions=True
)
```

这里设置`support_subquestions=True`会自动识别问题中的小题，并分别判断是否答对。否则会把整道题当做一个整体来看。

这一步如果没有识别小题，会得到`answer_match_result`一项；识别小题就还会得到 `correct_answer_num`, `total_subquestions`两项。注意识别小题时，只要有一道小题做错，`answer_match_result`就会是false。

`eval_result_path`会存入整体的正确率等信息（大题正确率，小题正确率），例如：
```json
[
  {
    "bench_name_or_prefix":"math-Qwen3-8B-Instruct",
    "total_samples":23584,
    "valid_samples":23571,
    "matched_samples":12281,
    "accuracy":0.5210215943,
    "empty_responses_count":13,
    "compare_method":"semantic",
    "total_subquestions":26380,
    "correct_subquestions":13807,
    "subquestion_accuracy":0.523388931
  }
]
```

然后根据`answer_match_result`的结果进行reject sampling，直到达到最大轮数或者不再有被reject的样本为止。最后会把curated的数据存入同一目录的`curated_vqa_with_cot.jsonl`当中。
