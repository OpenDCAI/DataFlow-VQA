在根目录下跑
```bash
python pipelines/vqa_extract_optimized_pipeline.py
```
可以extract VQA，注意进入代码中按照要求准备好输入的jsonl文件！
切割出的VQA会存到`output_dir`中的`vqa_filtered_qa_pairs.jsonl`中。

跑完extract后，可以直接用`bench_sampling.py`进行问题过滤和评测（对于没有answer的，会尝试从solution中提取answer）。

注意目前都需要在代码中修改输入输出路径。
