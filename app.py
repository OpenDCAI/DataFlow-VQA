import os
import sys
import re
import shutil
import tempfile
import traceback

import gradio as gr

# Ensure the repo root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_curation(
    input_file,
    api_url: str,
    api_key: str,
    model_name: str,
    max_workers: int,
    progress=gr.Progress(track_tqdm=True),
):
    if input_file is None:
        return None, "请先上传输入的 JSONL 文件。"
    if not api_key.strip():
        return None, "请填写 API Key。"

    # Inject the key so DataFlow's APILLMServing_request picks it up
    os.environ["DF_API_KEY"] = api_key.strip()

    # Use a dedicated temp workspace so parallel runs don't collide
    workspace = tempfile.mkdtemp(prefix="dataflow_vqa_")
    cache_dir = os.path.join(workspace, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # curate_data.py hardcodes cache_path="./cache", so we work from workspace
    original_cwd = os.getcwd()
    try:
        os.chdir(workspace)

        # Late import after path & cwd are set up
        from pipelines.curate_data import DataCurationPipeline

        progress(0.05, desc="初始化 pipeline…")

        pipeline = DataCurationPipeline(
            input_file=input_file,
            api_url=api_url.rstrip("/"),
            model_name=model_name,
            max_workers=int(max_workers),
        )
        pipeline.compile()

        progress(0.15, desc="正在运行 pipeline（可能需要几分钟）…")
        pipeline.forward()

        # Locate the highest-numbered step file
        step_files = [
            f for f in os.listdir(cache_dir)
            if re.match(r"curate_data_step\d+\.jsonl", f)
        ]
        if not step_files:
            return None, "Pipeline 运行完成，但未找到输出文件。请检查日志。"

        max_step = max(
            int(re.findall(r"curate_data_step(\d+)\.jsonl", f)[0])
            for f in step_files
        )
        output_path = os.path.join(cache_dir, f"curate_data_step{max_step}.jsonl")

        # Copy to a stable temp file so Gradio can serve it
        result_file = os.path.join(workspace, "curated_vqa.jsonl")
        shutil.copy(output_path, result_file)

        progress(1.0, desc="完成！")
        return result_file, f"✅ 完成！共执行 {max_step} 步，结果已保存为 curated_vqa.jsonl。"

    except Exception:
        tb = traceback.format_exc()
        return None, f"❌ 运行出错：\n```\n{tb}\n```"
    finally:
        os.chdir(original_cwd)


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="DataFlow-VQA · 数据清洗 Demo",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
# 🔬 DataFlow-VQA — 数据清洗 Pipeline Demo

将从 PDF 中提取的原始 VQA 数据（`raw_vqa.jsonl`）通过多步 LLM 清洗，输出高质量的 `curated_vqa.jsonl`。

**清洗步骤：** 子问题拆分 → 题型分类过滤 → 答案提取 → 填空补全 → 文本清理 → QA 质量过滤

> 注意：所有 LLM 调用均通过您提供的 API 完成，本 Space 不存储任何数据或密钥。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 输入")
            input_file = gr.File(
                label="上传输入 JSONL 文件（raw_vqa.jsonl）",
                file_types=[".jsonl"],
            )
            gr.Markdown("### ⚙️ API 配置")
            api_url = gr.Textbox(
                label="API Base URL（不含 /chat/completions）",
                value="https://api.openai.com/v1",
                placeholder="https://api.openai.com/v1",
            )
            api_key = gr.Textbox(
                label="API Key",
                placeholder="sk-...",
                type="password",
            )
            model_name = gr.Textbox(
                label="模型名称",
                value="gpt-4o-mini",
                placeholder="gpt-4o-mini / gemini-2.0-flash / deepseek-chat …",
            )
            max_workers = gr.Slider(
                label="并发 Worker 数量",
                minimum=1,
                maximum=50,
                value=5,
                step=1,
                info="HF Spaces 免费版资源有限，建议不超过 10",
            )
            run_btn = gr.Button("▶ 开始清洗", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📤 输出")
            status_box = gr.Textbox(
                label="运行状态",
                interactive=False,
                lines=6,
                placeholder="点击「开始清洗」后，状态信息将显示在这里…",
            )
            output_file = gr.File(
                label="下载清洗结果（curated_vqa.jsonl）",
                interactive=False,
            )

    gr.Markdown(
        """
---
**输入格式**：每行一个 JSON 对象，需包含 `question`、`answer`、`solution` 字段。

**支持的 API**：任何 OpenAI 兼容接口，包括 OpenAI、Google Gemini（via proxy）、DeepSeek、vLLM 等。

**项目地址**：[OpenDCAI/DataFlow-VQA](https://github.com/OpenDCAI/DataFlow-VQA)
        """
    )

    run_btn.click(
        fn=run_curation,
        inputs=[input_file, api_url, api_key, model_name, max_workers],
        outputs=[output_file, status_box],
    )

if __name__ == "__main__":
    demo.launch()
