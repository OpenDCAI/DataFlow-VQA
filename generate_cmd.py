import os
import re

root = "/data1/VQA_ready_data"  # 当前目录
output = "run_all.sh"

pattern = re.compile(r"(.+?)_(\d+)$")

warnings = []
cmds = []


def folder_has_suffix(folder, suffix):
    for fname in os.listdir(folder):
        if fname.endswith(suffix):
            return True
    return False


# 扫描所有 subject_x 目录
for name in os.listdir(root):
    folder = os.path.join(root, name)
    if not os.path.isdir(folder):
        continue

    m = pattern.match(name)
    if not m:
        continue

    subject, num = m.group(1), int(m.group(2))

    has10 = folder_has_suffix(folder, "step10.json")
    has11 = folder_has_suffix(folder, "step11.json")

    # 逐版本判断是否可以跑
    if has10 and not has11:
        cmds.append(f"python bench_sampling.py {subject} {num}")

# 写入输出脚本
with open(output, "w", encoding="utf-8") as f:
    for c in cmds:
        f.write(c + "\n")

print(f"写入 {output} 完成，共 {len(cmds)} 条命令")
if warnings:
    print("\n".join(warnings))
