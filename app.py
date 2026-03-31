import os
import sys
import re
import json
import shutil
import tempfile
import traceback

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_vqa_extraction(
    pdf_files,
    task_name: str,
    api_url: str,
    llm_api_key: str,
    mineru_api_key: str,
    model_name: str,
    max_workers: int,
    progress=gr.Progress(track_tqdm=True),
):
    if not pdf_files:
        return None, "请至少上传一个 PDF 文件。"
    if not llm_api_key.strip():
        return None, "请填写 LLM API Key。"
    if not mineru_api_key.strip():
        return None, "请填写 MinerU API Key。"
    if not task_name.strip():
        task_name = "task1"

    os.environ["DF_API_KEY"] = llm_api_key.strip()
    os.environ["MINERU_API_KEY"] = mineru_api_key.strip()

    workspace = tempfile.mkdtemp(prefix="dataflow_vqa_")
    cache_dir = os.path.join(workspace, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    original_cwd = os.getcwd()
    try:
        os.chdir(workspace)

        # Copy uploaded PDFs into workspace and build input JSONL
        pdf_paths = []
        for i, f in enumerate(pdf_files):
            dst = os.path.join(workspace, f"input_{i}.pdf")
            shutil.copy(f, dst)
            pdf_paths.append(dst)

        input_jsonl = os.path.join(workspace, "input.jsonl")
        with open(input_jsonl, "w") as fout:
            entry = {
                "input_pdf_paths": pdf_paths if len(pdf_paths) > 1 else pdf_paths[0],
                "name": task_name.strip(),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

        progress(0.05, desc="初始化 pipeline…")

        from pipelines.vqa_extract_optimized_pipeline import PDF_VQA_extract_optimized_pipeline

        pipeline = PDF_VQA_extract_optimized_pipeline(
            input_file=input_jsonl,
            api_url=api_url.rstrip("/"),
            model_name=model_name,
            max_workers=int(max_workers),
        )
        pipeline.compile()

        progress(0.15, desc="调用 MinerU 解析 PDF（可能需要几分钟）…")
        pipeline.forward()

        # Locate the highest-numbered step file
        step_files = [
            f for f in os.listdir(cache_dir)
            if re.match(r"vqa_step\d+\.jsonl", f)
        ]
        if not step_files:
            return None, "Pipeline 运行完成，但未找到输出文件，请检查日志。"

        max_step = max(
            int(re.findall(r"vqa_step(\d+)\.jsonl", f)[0])
            for f in step_files
        )
        max_step_file = os.path.join(cache_dir, f"vqa_step{max_step}.jsonl")

        # Extract vqa_pair items → raw_vqa.jsonl
        result_file = os.path.join(workspace, "raw_vqa.jsonl")
        count = 0
        with open(max_step_file) as f_in, open(result_file, "w") as f_out:
            for line in f_in:
                data = json.loads(line)
                qa_item = data.get("vqa_pair")
                if not qa_item:
                    continue
                name = data.get("name", task_name)
                out = {"name": name, **qa_item}
                if not out.get("solution"):
                    out["solution"] = out.get("answer", "")
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                count += 1

        progress(1.0, desc="完成！")
        return result_file, f"✅ 完成！共提取 {count} 条 QA 对，已保存为 raw_vqa.jsonl。"

    except Exception:
        tb = traceback.format_exc()
        return None, f"❌ 运行出错：\n```\n{tb}\n```"
    finally:
        os.chdir(original_cwd)


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="DataFlow-VQA · PDF 提取 Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🔬 DataFlow-VQA — 从 PDF 提取 VQA 数据

上传教材或试卷 PDF，自动用 [MinerU](https://mineru.net) 解析版面、再用 LLM 提取结构化 QA 对，输出 `raw_vqa.jsonl`。

**流程：** PDF 合并 → MinerU 解析 → LLM 提取 QA → 后处理输出

> 所有 API 调用均通过您提供的密钥完成，本 Space 不存储任何数据或密钥。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 上传 PDF")
            pdf_files = gr.File(
                label="上传 PDF（单文件：题答混排；两文件：第1个为题，第2个为答案）",
                file_types=[".pdf"],
                file_count="multiple",
            )
            task_name = gr.Textbox(
                label="任务名称（用于目录命名）",
                value="task1",
                placeholder="task1",
            )

            gr.Markdown("### ⚙️ LLM API 配置")
            api_url = gr.Textbox(
                label="API Base URL",
                value="https://generativelanguage.googleapis.com/v1beta/openai/",
                placeholder="https://api.openai.com/v1",
            )
            llm_api_key = gr.Textbox(
                label="LLM API Key (DF_API_KEY)",
                placeholder="sk-... / AIzaSy...",
                type="password",
            )
            model_name = gr.Textbox(
                label="模型名称（推荐强推理模型）",
                value="gemini-2.5-pro",
                placeholder="gemini-2.5-pro / gpt-4o / deepseek-r1",
            )

            gr.Markdown("### 🏗️ MinerU API 配置")
            mineru_api_key = gr.Textbox(
                label="MinerU API Key (MINERU_API_KEY)",
                placeholder="sk2-...",
                type="password",
                info="在 https://mineru.net/apiManage/token 申请",
            )

            max_workers = gr.Slider(
                label="并发 Worker 数",
                minimum=1, maximum=30, value=5, step=1,
            )
            run_btn = gr.Button("▶ 开始提取", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📤 输出")
            status_box = gr.Textbox(
                label="运行状态",
                interactive=False,
                lines=8,
                placeholder="点击「开始提取」后状态信息显示在这里…",
            )
            output_file = gr.File(
                label="下载提取结果（raw_vqa.jsonl）",
                interactive=False,
            )

    gr.Markdown(
        """
---
**输入说明**
- 单 PDF：题目和答案在同一文件中交织排布
- 双 PDF：第一个上传题目 PDF，第二个上传答案 PDF

**输出格式** `raw_vqa.jsonl`，每行含 `name / question / answer / solution / images` 等字段，可直接送入 Stage 2（数据清洗）。

**项目地址**：[OpenDCAI/DataFlow-VQA](https://github.com/OpenDCAI/DataFlow-VQA)
        """
    )

    run_btn.click(
        fn=run_vqa_extraction,
        inputs=[pdf_files, task_name, api_url, llm_api_key, mineru_api_key, model_name, max_workers],
        outputs=[output_file, status_box],
    )

if __name__ == "__main__":
    demo.launch()
