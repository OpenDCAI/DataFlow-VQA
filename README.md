# DataFlow-VQA

**[中文文档](README_zh.md)**

A pipeline for extracting, curating, and generating chain-of-thought (CoT) data from PDF textbooks and exam papers.

[📄Full Paper with Appendices](./FlipVQA_full.pdf) [🤗Dataset](https://huggingface.co/datasets/OpenDCAI/FlipVQA) [🤗FlipVQA-Miner Demo](https://huggingface.co/spaces/aaron1141/FlipVQA)

## Overview
![DataFlow-VQA overview](static/overview_2.png)
DataFlow-VQA processes PDF documents through three sequential stages:

- Stage1 (**Section 3.1: VQA Extraction**): Parses PDFs using [MinerU](https://github.com/opendatalab/MinerU) for document layout analysis, then uses an LLM to extract structured question-answer pairs with images.
- Stage2 (**Section 3.2.1 to Section 3.2.5: Data Curation**): Filters and cleans the extracted QA pairs — splits sub-questions, classifies question types, extracts concise answers, and removes low-quality items.
- Stage3 (**Section 3.2.6: CoT Generation**): Generates chain-of-thought reasoning via reject sampling — an LLM generates answers, which are verified against ground truth, and incorrect ones are retried.



## Installation

This project is built on top of [DataFlow](https://github.com/OpenDCAI/DataFlow). Clone and install it first:

```shell
git clone https://github.com/OpenDCAI/DataFlow.git
cd DataFlow
pip install -e ".[pdf2vqa]"
```

Then clone this repository:

```shell
git clone <this-repo-url>
cd DataFlow-VQA
```

## Configuration

### API Keys

Two API keys are required:

- `DF_API_KEY`: API key for the LLM service (OpenAI, Google Gemini, DeepSeek, etc.)
- `MINERU_API_KEY`: API key for [MinerU](https://mineru.net/apiManage/token) document layout parsing

```shell
export DF_API_KEY="sk-xxxxx"
export MINERU_API_KEY="sk2-xxxxx"
```

### LLM Endpoint

Each pipeline accepts `--api_url` and `--model` arguments. Any [OpenAI-compatible API](https://platform.openai.com/docs/api-reference) endpoint is supported, including OpenAI, Google Gemini (via proxy), DeepSeek, and others.

Provide the **base URL** without `/chat/completions` (e.g. `https://api.openai.com/v1`).

---

## Stage 1: VQA Extraction

### Input Format

Create a JSONL file where each line describes one PDF extraction task:

```jsonl
{"input_pdf_paths": "./examples/VQA/questionextract_test.pdf", "name": "math1"}
{"input_pdf_paths": ["./examples/VQA/math_question.pdf", "./examples/VQA/math_answer.pdf"], "name": "math2"}
```

- `input_pdf_paths`: A single PDF (questions and answers interleaved) or a list of two or more PDFs (questions before answers).
- `name`: A unique identifier for this task (used for directory naming and caching).

### Run

```bash
python -m pipelines.vqa_extract_optimized_pipeline \
    --input_file ./examples/VQA/vqa_extract_test.jsonl \
    --output_dir ./output \
    --api_url https://generativelanguage.googleapis.com/v1beta/openai/ \
    --model gemini-2.5-pro
```

**Important:** We recommend using a strong powerful model here. Weak models like `gpt-5-mini` might perform bad.

### Output

- `{output_dir}/raw_vqa.jsonl`: Extracted QA pairs with image references
- `{output_dir}/{name}/vqa_images/`: Extracted images
- `cache/{name}/extracted_vqa.jsonl`, `merged_qa_pairs.jsonl`, `merged_qa_pairs.md`: Per-task intermediate files

Each QA item contains:

```json
{
  "question": "Compute $x$ such that $x^2 - 1 = 0$.",
  "answer": "$x = 1$ or $x = -1$",
  "solution": "Factor as $(x-1)(x+1)=0$.",
  "label": 1,
  "question_chapter_title": "Chapter 1: Quadratic Equations",
  "answer_chapter_title": "Chapter 1: Quadratic Equations",
  "image_basedir": "/path/to/your/images"
}
```

### Note

**We also support using a local MinerU deployment**: Replace `FileOrURLToMarkdownConverterAPI` with `FileOrURLToMarkdownConverterLocal` or `FileOrURLToMarkdownConverterFlash` in `pipelines/vqa_extract_optimized_pipeline.py`:

```python
# Original opendatalab local version
self.mineru_executor = FileOrURLToMarkdownConverterLocal(
    intermediate_dir="intermediate",
    mineru_model_path="path/to/mineru/model",
)

# Accelerated version (Flash)
self.mineru_executor = FileOrURLToMarkdownConverterFlash(
    intermediate_dir="intermediate",
    mineru_model_path="path/to/mineru/model",
    batch_size=4,
    replicas=1,
    num_gpus_per_replica=1,
    engine_gpu_util_rate_to_ray_cap=0.9,
)
```

See [DataFlow's MinerU operators](https://github.com/OpenDCAI/DataFlow/blob/main/dataflow/operators/knowledge_cleaning/generate/mineru_operators.py) for full parameter documentation.

<details>
<summary>Pipeline details</summary>

The extraction pipeline runs six steps:

1. **PDF Merging** (`PDF_Merger`): If multiple PDFs are provided, merges them into one.
2. **Document Layout Parsing** (`FileOrURLToMarkdownConverterAPI`): Calls the MinerU API to produce structured JSON layout tokens and page images.
3. **Layout Preprocessing** (`MinerU2LLMInputOperator`): Flattens list items and re-indexes IDs to prepare LLM-ready input.
4. **LLM Extraction** (`ChunkedPromptedGenerator`): Chunks the layout JSON (max 128k tokens per chunk) and calls the LLM with `QAExtractPrompt` to extract QA pairs as structured XML.
5. **Output Parsing** (`LLMOutputParser`): Parses the XML response into JSONL and copies images to `vqa_images/`.
6. **QA Merging** (`QA_Merger`): For separated question/answer PDFs, matches question and answer blocks by chapter title and question number.
This operator includes a `strict_title_match` parameter: When set to True, the operator performs an exact string match on chapter titles. Otherwise, the operator attempts to extract Chinese or English sequence numbers from the titles for matching.

</details>

---

## Stage 2: Data Curation

```bash
python -m pipelines.curate_data \
    --input_file ./output/raw_vqa.jsonl \
    --api_url https://api.openai.com/v1 \
    --model gpt-5-mini
```

Output is saved as `curated_vqa.jsonl` in the same directory as `--input_file`.

<details>
<summary>Pipeline details</summary>

Four sequential steps:

**1. Sub-question Splitting**

Questions with multiple independent parts (e.g. (a), (b), (c)) are split into separate items. Each sub-question is paired with its corresponding sub-answer and sub-solution. Items where the question or both answer and solution are empty are discarded.

Sub-questions that are context-sensitive (e.g. (b) uses the result of (a)) will not be split into separate items.

Adds field: `split_qa`

**2. Question Type Classification**

Each question is classified as one of: `Calculation`, `Proof`, `Explanation`, `Fill-in`, `Multiple-choice`, `Sketching`, `Other`.

By default, only `Calculation`, `Fill-in`, and `Multiple-choice` are retained. To change this, edit the `filter_rules` list in `DataCurationPipeline.__init__`.

Adds fields: `type`, `type_reason`

**3. Answer Extraction**

Extracts a concise final answer from the `solution` field and writes it to `answer`. Items that already have a non-empty `answer` are skipped (set `overwrite=True` in `AnswerExtractionOperator` to override).

**4. QA Filtering**

Removes items based on the following criteria:

- The question must pose a clear, specific problem suitable for an exam. Examples, statements without questions, and open-ended discussions are rejected.
- The answer must directly address the question.
- The question and answer must be self-contained, without relying on external references or omitted context.

Adds fields: `filter_result`, `filter_reason`

</details>

---

## Stage 3: Generate CoT

The answer model and judge model can use different API endpoints and API keys, which is useful when the answer model is a self-hosted open-source VLM (e.g. Qwen3-VL served via vLLM) and the judge model is a commercial API.

Use `--answer_api_key_env` / `--judge_api_key_env` to specify which environment variable holds the API key for each model (default: `DF_API_KEY` for both).

```bash
# Example: self-hosted Qwen3-VL for answers, OpenAI for judging
export VLLM_API_KEY="token-xxxx"   # or leave empty if your vLLM server needs no key
export DF_API_KEY="sk-xxxx"

python -m pipelines.generate_cot \
    --input_file ./output/curated_vqa.jsonl \
    --max_retries 5 \
    --answer_api_url https://your-vllm-server/v1 \
    --answer_model qwen3-vl-235b-thinking \
    --answer_api_key_env VLLM_API_KEY \
    --judge_api_url https://api.openai.com/v1 \
    --judge_model gpt-5-mini \
    --judge_api_key_env DF_API_KEY
```

Output is saved as `curated_vqa_with_cot.jsonl` in the same directory as `--input_file`.

<details>
<summary>Pipeline details</summary>

Uses reject sampling over up to `max_retries` rounds:

**1. Answer Generation** (`VQAReasoningAnswerGenerator`)

The LLM generates a step-by-step answer. Set `skip_text_only=True` in `RejectSamplingPipeline` to process only VQA items (questions containing images); set to `False` to process all items. Generated answer stored in `generated_cot`.

**2. Thinking Cleanup**

Strips `<think>...</think>` content from the generated answer to reduce verification cost. The cleaned answer is stored in `llm_short_answer`. Assumes the model outputs `<think>THINK</think>ANSWER` or `THINK</think>ANSWER`.

**3. Answer Verification** (`BenchDatasetEvaluatorQuestion`)

Compares `llm_short_answer` against the ground truth `answer` using semantic LLM evaluation (with 5% numerical tolerance). Items that pass are marked `answer_match_result = True` and skipped in subsequent rounds.

Set `support_subquestions=True` to evaluate each sub-question independently; `answer_match_result` is `False` if any sub-question is wrong.

Evaluation statistics (overall accuracy, sub-question accuracy) are saved to `./cot_cache/eval_results.jsonl`:

```json
{
  "total_samples": 23584,
  "matched_samples": 12281,
  "accuracy": 0.521,
  "total_subquestions": 26380,
  "correct_subquestions": 13807,
  "subquestion_accuracy": 0.523
}
```

</details>

---

## Examples

Sample PDFs and input JSONL are provided in `examples/VQA/`:

```
examples/VQA/
├── vqa_extract_test.jsonl    # Example input for Stage 1
├── questionextract_test.pdf  # Single PDF with interleaved Q&A
├── math_question.pdf         # Questions PDF (for separated Q&A demo)
└── math_answer.pdf           # Answers PDF (for separated Q&A demo)
```

To run the full pipeline on the examples:

```bash
# Stage 1: Extract
python -m pipelines.vqa_extract_optimized_pipeline \
    --input_file ./examples/VQA/vqa_extract_test.jsonl \
    --output_dir ./output \
    --api_url https://generativelanguage.googleapis.com/v1beta/openai/ \
    --model gemini-2.5-pro

# Stage 2: Curate
python -m pipelines.curate_data \
    --input_file ./output/raw_vqa.jsonl \
    --api_url https://api.openai.com/v1 \
    --model gpt-5-mini

# Stage 3: Generate CoT
# Example: self-hosted Qwen3-VL for answers, OpenAI for judging
export VLLM_API_KEY="token-xxxx"   # or leave empty if your vLLM server needs no key
export DF_API_KEY="sk-xxxx"

python -m pipelines.generate_cot \
    --input_file ./output/curated_vqa.jsonl \
    --max_retries 5 \
    --answer_api_url https://your-vllm-server/v1 \
    --answer_model qwen3-vl-235b-thinking \
    --answer_api_key_env VLLM_API_KEY \
    --judge_api_url https://api.openai.com/v1 \
    --judge_model gpt-5-mini \
    --judge_api_key_env DF_API_KEY
```

## Note
The implementation in this repository is only for running a demo at small scale. If you wish to run the pipeline on large number of books, you will probably need features [Checkpoint Resume](https://opendcai.github.io/DataFlow-Doc/en/guide/resume/) and [Batched Inference](https://opendcai.github.io/DataFlow-Doc/en/guide/batch/).

## License

This project is licensed under the [Apache License 2.0](LICENSE).
