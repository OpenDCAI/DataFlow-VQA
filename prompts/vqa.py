from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

@PROMPT_REGISTRY.register()
class QAExtractPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, subject: str = "math") -> str:
        PROMPT = f"""
        You are an expert in {subject}. You are given a json file. Your task is to segment the content, insert images tags, and extract labels:
1. Every json item has an "id" field. Your main task is to output this field.
2. You need to segment the content into multiple `<qa_pair>`…`</qa_pair>` blocks, each containing a question and its corresponding answer with solution.
3. If the problem or answer is not complete, omit them.
4. You need to put the images id into proper positions. You could look at the caption or context to decide where to put the image tags.
5. You will also need to extract the chapter title and each problem's label/number from the text.
6. You only need to output "id" field for chapters, questions and solutions. DO NOT OUTPUT ORIGINAL TEXT. Use ',' to separate different ids.
7. However, use original labels/numbers for labels, and use original numbers for answers. DO NOT output "id" field for labels and answers. You will need to extract them from the text.
"""
        PROMPT +=f"""
Strict extraction rules:
** About questions and answers/solutions **
- Preserve each problem’s original label/number, such as "例1", "Example 3", "习题1", "11". Do not include the period after the number. Use Arabic numerals only. For example, if the label is "例一", convert it to "例1". If the label is "IV", convert it to "4". 
- If the full label is "三、16", keep only "16". If the full label is "5.4", keep only "4".
- If there are multiple sub-questions (such as "(1)", "(a)") under one main question, always put them together in the same `<qa_pair>`…`</qa_pair>` block.
- If a question and its answer/solution are contiguous, wrap them together as a single `<qa_pair>`…`</qa_pair>` block, e.g.:
  `<qa_pair><label>例1</label><question>…</question><answer>…</answer><solution>…</solution></qa_pair>`
- If only questions or only answers with solutions appear, wrap each question or answer with solution in a `<qa_pair>`…`</qa_pair>` block with the missing part left empty. For example, if only questions appear:
  `<qa_pair><label>例1</label><question>…</question><answer></answer><solution></solution></qa_pair>`
- If multiple questions and solutions appear, wrap each question/solution pair in its own `<qa_pair>`…`</qa_pair>` block.
- If you do not see the full solution, only extract the short answer and leave the solution empty. YOU MUST KEEP QUESTIONS WITH ONLY SHORT ANSWERS !!!
** About chapter/section titles **
- Always enclose qa pairs in a `<chapter>`…`</chapter>` block, where <title>MAIN_TITLE</title> is the chapter title or section title.
- Normally, chapter/section titles appear before the questions/answers in an independent json item.
- There could be multiple `<chapter>`…`</chapter>` blocks if multiple chapters/sections exist. 
- **Any titles followed by a question/answer whose label/number is not 1, or with a score after the title such as `一、选择题（每小题3分，共15分）`, should NOT be extracted.**
- Do not use nested titles.
- Leave the title blank if there is no chapter title. 
** About figures/diagrams **
- Whenever the question or answer/solution refers to a figure or diagram, record its "id" in question/answer/solution just like other text content.
- You MUST include all images referenced in the question/answer/solution.


If no qualifying content is found, output:
<empty></empty>

Output format (all tags run together, no extra whitespace or newlines except between entries):
<chapter><title>MAIN_TITLE_ID</title>
<qa_pair><label>…</label><question>QUESTION_IDS</question>
<answer>ANSWER(EXTRACTED FROM SOLUTION)</answer><solution>SOLUTION_IDS</solution></qa_pair>
<qa_pair><label>…</label><question>QUESTION_IDS</question>
<answer>ANSWER(EXTRACTED FROM SOLUTION)</answer><solution></solution></qa_pair>
</chapter>
<chapter><title>MAIN_TITLE</title>
<qa_pair><label>…</label><question>QUESTION_IDS</question>
<answer>ANSWER(EXTRACTED FROM SOLUTION)</answer><solution>SOLUTION_IDS</solution></qa_pair>
</chapter>


Example:
<chapter><title>1</title>
<qa_pair><label>例1</label><question>2,3</question>
<answer>4/5</answer><solution>5,6,7</solution></qa_pair>
<qa_pair><label>例2</label><question>8,9,10</question>
<answer>3.14</answer><solution></solution></qa_pair>
</chapter>
<chapter><title>12</title>
<qa_pair><label>例1</label><question>13,14</question>
<answer>2^6</answer><solution>16</solution></qa_pair>
</chapter>

Please now process the provided json and output your result.
"""
        return PROMPT