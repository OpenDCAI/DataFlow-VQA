在根目录下跑
```bash
python pipelines/vqa_extract_optimized_pipeline.py
```
可以extract VQA，注意进入代码中按照要求准备好输入的jsonl文件！
切割出的VQA会存到`output_dir`中的`vqa_filtered_qa_pairs.jsonl`中。

跑完extract后，可以直接用`bench_sampling.py`进行问题过滤和评测（对于没有answer的，会尝试从solution中提取answer）。

注意目前都需要在代码中修改输入输出路径。

## VQA Extraction
这部分教程参考https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd 的1.10

## Bench Sampling
代码在`bench_sampling.py`当中。

主函数（可以命令行跑，应该不太要自己改）：
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BenchSamplingPipeline")
    parser.add_argument("name", nargs="?", default="pde", help="dataset name (default: pde)")
    parser.add_argument("end", nargs="?", type=int, default=3, help="range end for i (uses range(1, end + 1), default: 3)")
    args = parser.parse_args()
    
    # Example usage:
    # python bench_sampling.py pde 3

    name = args.name

    for i in range(1, args.end + 1):
    
        first_entry_file_name=f"/data1/VQA_ready_data/{name}_{i}/vqa_filtered_qa_pairs.jsonl"
        cache_path = f"/data1/VQA_ready_data/{name}_{i}"
        file_name_prefix = f"{name}_{i}"
        
        model = BenchSamplingPipeline(first_entry_file_name, cache_path, file_name_prefix)
        model.forward()
```

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

## Rollout
这部分代码在`pipelines/vqa_rollout.py`当中。

主函数：
```python
first_entry_file_name=f"/data1/VQA_ready_data/all_vqa.jsonl"
cache_path = f"./rollout_cache/all_math_vqa_only"
file_name_prefix = f"math-Qwen3-8B-Instruct"
eval_result_path = f"./rollout_cache/all_math/eval_results.jsonl"
input_image_default_basedir = f"./"

model = BenchSamplingPipeline(first_entry_file_name, cache_path, file_name_prefix, eval_result_path)
model.forward(input_image_default_basedir)
```
- `first_entry_file_name`: 输入文件，只要有question和answer两项即可。对于vqa，最好还要有`image_basedir`指明图片存储的根目录，会拼接到`!(CAPTION)[REL_PATH]`里面。
  **收集上面bench sampling过滤出来的所有qa并添加image_basedir的代码参考`utils/collect_all_vqa.py`**
- `cache_path`: 中间结果存储路径。
- `file_name_prefix`: 中间结果存储文件名前缀（`PREFIX_step_x`）。
- `eval_result_path`: 模型正确率存储路径。
- `input_image_default_basedir`: 默认图片路径，前面如果设置了`image_basedir`，这里就可以随便设置。


这部分分为5步：

- 复制题目，比如要rollout 32次就复制32份题目。`self.dup_operator = PandasOperator(process_fn=[ make_rollout_dup_fn(32) ])`
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
- 收集每道题的正确率：`self.agg_operator = PandasOperator(process_fn=[ make_rollout_aggregate_fn() ])`

  之前每道题rollout了若干次，现在会把每道题的正确率收集起来，写入`rollout_num`和`accuracy`两项中。

跑完上述内容后，完整结果示例：
```json
{
    "question":"如图2.8所示的 RL 电路，试求：当开关 $S_{1}$ 合上 10 s 后，电感 $L$ 上的电流。 ![图2.8 RL电路](question_images\/04ce3abe2b816fcfa1f116c1a6459212ed270a74d803f31da9701038a969ef99.jpg)",
    "question_chapter_title":null,
    "answer_chapter_title":null,
    "label":"3",
    "answer":"I(10 s) = 5(1 - e^{-50}) \\approx 5 A.",
    "solution":"解（1）设图2.8的电路上电流为  $I(t)$  ，当开关  $S_{1}$  合上时电路方程为\n\n$$\n\\frac {\\mathrm {d} I}{\\mathrm {d} t} + \\frac {R _ {1}}{L} I = \\frac {E}{L}, \\text {即} \\frac {\\mathrm {d} I}{\\mathrm {d} t} = - 5 I + 2 5.\n$$\n\n有解  $I(t) = 5 - c\\mathrm{e}^{-5t}$ , 其中  $c$  为任意常数. 因刚合上时  $I(0) = 0$  得  $c = 5$ , 解为  $I(t) = 5(1 - \\mathrm{e}^{-5t})$ .  $S_{1}$  合上  $10\\mathrm{s}$  后\n\n$$\nI (t) = 5 \\left(1 - e ^ {- 5 0}\\right) \\approx 5 A.\n$$",
    "split_qa":"[\n  {\n    \"sub_id\": 1,\n    \"sub_question\": \"如图2.8所示的 RL 电路，试求：当开关 $S_{1}$ 合上 10 s 后，电感 $L$ 上的电流。 ![图2.8 RL电路](question_images\/04ce3abe2b816fcfa1f116c1a6459212ed270a74d803f31da9701038a969ef99.jpg)\",\n    \"sub_answer\": \"I(10 s) = 5(1 - e^{-50}) \\\\approx 5 A.\",\n    \"sub_solution\": \"解（1）设图2.8的电路上电流为  $I(t)$  ，当开关  $S_{1}$  合上时电路方程为\\n\\n$$\\n\\\\frac {\\\\mathrm {d} I}{\\\\mathrm {d} t} + \\\\frac {R _ {1}}{L} I = \\\\frac {E}{L}, \\\\text {即} \\\\frac {\\\\mathrm {d} I}{\\\\mathrm {d} t} = - 5 I + 2 5.\\n$$\\n\\n有解  $I(t) = 5 - c\\\\mathrm{e}^{-5t}$ , 其中  $c$  为任意常数. 因刚合上时  $I(0) = 0$  得  $c = 5$ , 解为  $I(t) = 5(1 - \\\\mathrm{e}^{-5t})$ .  $S_{1}$  合上  $10\\\\mathrm{s}$  后\\n\\n$$\\nI (t) = 5 \\\\left(1 - e ^ {- 5 0}\\\\right) \\\\approx 5 A.\\n$$\"\n  },\n  {\n    \"sub_id\": 2,\n    \"sub_question\": \"如图2.8所示的 RL 电路，试求：$S_{1}$ 合上 10 s 后再将 $S_{2}$ 合上，求 $S_{2}$ 合上 20 s 后电感 $L$ 上的电流。 ![图2.8 RL电路](question_images\/04ce3abe2b816fcfa1f116c1a6459212ed270a74d803f31da9701038a969ef99.jpg)\",\n    \"sub_answer\": \"I(20 s) = 7.5 - 2.5 e^{-66.6} \\\\approx 7.5 A.\",\n    \"sub_solution\": \"(2)  $S_{1}$  合上  $10\\\\mathrm{~s~}$  后再将  $S_{2}$  合上，令  $\\\\frac{1}{R} = \\\\frac{1}{R_1} +\\\\frac{1}{R_2} = \\\\frac{3}{20}$  此时电路方程变为  $\\\\frac{\\\\mathrm{d}I}{\\\\mathrm{d}t} = -\\\\frac{10}{3} I + 25.$  方程的解为  $I(t) = 7.5 + ce^{-{\\\\frac{10}{3}} t}.$  因\\n开始时  $I(0) = 5$  ，得解  $I(t) = 7.5 - 2.5\\\\mathrm{e}^{-\\\\frac{10}{3} t}$ .  $S_{2}$  合上  $20\\\\mathrm{s}$  后有  $I(t) = 7.5 - 2.5\\\\mathrm{e}^{-66.6} \\\\approx 7.5\\\\mathrm{A}$ .\"\n  }\n]",
    "question_type":"{\n  \"type\": \"Calculation\",\n  \"reason\": \"The problem asks for the numeric\/symbolic value of the inductor current after 10 s in an RL transient circuit and the provided answer is a computed expression and numerical approximation.\"\n}",
    "type":"Calculation",
    "type_reason":"The problem asks for the numeric\/symbolic value of the inductor current after 10 s in an RL transient circuit and the provided answer is a computed expression and numerical approximation.",
    "qa_judgement":"{\n  \"reason\": \"The item poses a clear calculational question (current in the inductor 10 s after closing the switch) and the provided numerical answer directly addresses that question. The image presumably supplies the circuit parameters needed, and a concise numeric result is acceptable for an exam-style problem.\",\n  \"judgement\": \"true\"\n}",
    "filter_result":"true",
    "filter_reason":"The item poses a clear calculational question (current in the inductor 10 s after closing the switch) and the provided numerical answer directly addresses that question. The image presumably supplies the circuit parameters needed, and a concise numeric result is acceptable for an exam-style problem.",
    "image_basedir":"\/data1\/VQA_ready_data\/ode_2",
    "subject":"ode_2",
    "chapter_title":"第二章 一阶微分方程的初等解法",
    "rollout_index":0,
    "llm_answer":"Solution:\n1. Analyze circuit configuration when switch $S_1$ is closed:\n   → Switch $S_1$ closes the branch containing resistor $R_1 = 10\\,\\Omega$, while $S_2$ remains open.\n   → The inductor $L = 2\\,\\text{H}$ is in series with $R_1$ and the voltage source $E = 50\\,\\text{V}$.\n\n2. Determine time constant $\\tau$ of RL circuit:\n   → Time constant formula: $\\tau = \\frac{L}{R}$\n   → Total resistance in the loop: $R = R_1 = 10\\,\\Omega$\n   → $\\tau = \\frac{2\\,\\text{H}}{10\\,\\Omega} = 0.2\\,\\text{s}$\n\n3. Derive current expression for RL circuit after switch closure:\n   → Current in RL circuit grows exponentially: $i(t) = I_{\\\\text{final}} \\left(1 - e^{-t\/\\tau}\\right)$\n   → Steady-state final current: $I_{\\\\text{final}} = \\frac{E}{R} = \\frac{50\\,\\text{V}}{10\\,\\Omega} = 5\\,\\text{A}$\n\n4. Calculate current at $t = 10\\,\\text{s}$:\n   → Substitute into current equation: $i(10) = 5 \\left(1 - e^{-10\/0.2}\\right) = 5 \\left(1 - e^{-50}\\right)$\n   → Since $e^{-50}$ is extremely small ($≈ 1.93×10^{-22}$), we have:\n     $i(10) ≈ 5 \\times (1 - 0) = 5\\,\\text{A}$\n\nVerification:\n→ Check if 10s is much greater than $\\tau$: $t = 10\\,\\text{s} >> \\tau = 0.2\\,\\text{s}$, so exponential term decays nearly completely.\n→ Result confirms approach: current approaches steady state.\n\n$\\boxed{5}$",
    "llm_short_answer":"Solution:\n1. Analyze circuit configuration when switch $S_1$ is closed:\n   → Switch $S_1$ closes the branch containing resistor $R_1 = 10\\,\\Omega$, while $S_2$ remains open.\n   → The inductor $L = 2\\,\\text{H}$ is in series with $R_1$ and the voltage source $E = 50\\,\\text{V}$.\n\n2. Determine time constant $\\tau$ of RL circuit:\n   → Time constant formula: $\\tau = \\frac{L}{R}$\n   → Total resistance in the loop: $R = R_1 = 10\\,\\Omega$\n   → $\\tau = \\frac{2\\,\\text{H}}{10\\,\\Omega} = 0.2\\,\\text{s}$\n\n3. Derive current expression for RL circuit after switch closure:\n   → Current in RL circuit grows exponentially: $i(t) = I_{\\\\text{final}} \\left(1 - e^{-t\/\\tau}\\right)$\n   → Steady-state final current: $I_{\\\\text{final}} = \\frac{E}{R} = \\frac{50\\,\\text{V}}{10\\,\\Omega} = 5\\,\\text{A}$\n\n4. Calculate current at $t = 10\\,\\text{s}$:\n   → Substitute into current equation: $i(10) = 5 \\left(1 - e^{-10\/0.2}\\right) = 5 \\left(1 - e^{-50}\\right)$\n   → Since $e^{-50}$ is extremely small ($≈ 1.93×10^{-22}$), we have:\n     $i(10) ≈ 5 \\times (1 - 0) = 5\\,\\text{A}$\n\nVerification:\n→ Check if 10s is much greater than $\\tau$: $t = 10\\,\\text{s} >> \\tau = 0.2\\,\\text{s}$, so exponential term decays nearly completely.\n→ Result confirms approach: current approaches steady state.\n\n$\\boxed{5}$",
    "answer_match_result":true,
    "correct_answer_num":1,
    "total_subquestions":1,
    "response_evaluation":"{\n  \"reason\": \"There is a single question (current at t = 10 s). The current answer gives the same expression I(10) = 5(1 - e^{-50}) and the same numerical conclusion ≈ 5 A as the reference. The time constant and final current used match the reference, and the approximation e^{-50} ≈ 0 is correctly applied. Therefore the answer is semantically consistent with the reference.\",\n  \"judgement\": [\"true\"]\n}",
    "rollout_num":32,
    "accuracy":1.0
  }
```

## Evaluate
这部分代码在`pipelines/vqa_evaluate.py`当中，逻辑与rollout基本一致。

主函数不变，输入文件中，如果要根据accuracy进行难度分类，则要多一个`accuracy`项，比如可以直接用上面rollout出来的结果。

代码分为4步：

- 难度过滤：`self.difficulty_filter = GeneralFilter(filter_rules=[lambda df: df['accuracy'] <= 0.6])`，可以自己改。
- llm 回答：同上
- 清理llm answer中的thinking部分：同上
- llm verify：同上
