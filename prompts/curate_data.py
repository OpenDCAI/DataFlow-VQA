import json
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC, DIYPromptABC
from typing import Set
import string

@PROMPT_REGISTRY.register()
class SubQuestionSplitingPrompt(DIYPromptABC):          
    def __init__(self, f_str_template: str = "{input_text}", on_missing: str = "raise"):
        self.f_str_template ="""
        You are an educational question structure analysis assistant. Below is a composite question and its corresponding answer. Please split it into several independent sub-questions.
The requirements are as follows:

1. The question may contain multiple sub-questions (e.g., ①②③ or (a)(b), etc.); please accurately identify and split them one by one. Only split sub-questions with clear labels.
    Do not split implicit sub-questions (such as "What is the value of x and y?" or multiple question marks).
2. Each sub-question must be self-contained and answerable. If the original question contains contextual information, include it in each sub-question to preserve full meaning.
    If sub-questions are related (e.g., "① Find x. ② Using the value of x, find y."), do not split them; keep them as one sub-question.
3. If an answer or/and solution is provided, try to match each sub-question with its corresponding part of the answer or/and solution based on semantics.
4. If the original answer or/and solution contains LaTeX formulas, preserve them exactly as they appear.
5. If the original answer or/and solution is missing or cannot be clearly aligned, leave `"sub_answer"` or/and `"sub_solution"` as an empty string.
6. The output must be a valid JSON array, where each element contains:

   * `"sub_id"`: the index of the sub-question (an integer starting from 1)
   * `"sub_question"`: the complete text of the sub-question (or "ORIGINAL" if no splitting is needed)
   * `"sub_answer"`: the corresponding answer, empty string if unavailable (or "ORIGINAL" if no splitting is needed)
   * `"sub_solution"`: the corresponding solution, empty string if unavailable (or "ORIGINAL" if no splitting is needed)
   
[Important Notice]
1. In some questions, answers or solutions, there will be figures written as `![image](image_url)`. When splitting, please keep these figure references in the corresponding sub-questions, sub-answers, or sub-solutions as EXACTLY what they are.
2. If the question does not need to be split, return an array with a single element, simplified as: [{"sub_id": 1, "sub_question": "ORIGINAL", "sub_answer": "ORIGINAL", "sub_solution": "ORIGINAL"}]
    In this case, you only need to output "ORIGINAL" instead of the full text for sub_question, sub_answer, and sub_solution, so that we can save tokens.

## Example Input:

**Question:**  
A class has 40 students, including 25 boys and 15 girls. ![image](question_images/a284h5iuh38.jpg) ① Find the percentage of boys in the class. ② Find the percentage of girls in the class.

**Answer:**  
① 62.5%. ② 37.5%.

**Solution:**  
Percentage of boys = (25/40) * 100 = 62.5%, percentage of girls = (15/40) * 100 = 37.5%.
----------------------------------------------

## Example Output:

```json
[
  {
    "sub_id": 1,
    "sub_question": "A class has 40 students, including 25 boys and 15 girls. ![image](question_images/a284h5iuh38.jpg) Find the percentage of boys in the class.",
    "sub_answer": "62.5%.",
    "sub_solution": "Percentage of boys = (25/40) * 100 = 62.5%."
  },
  {
    "sub_id": 2,
    "sub_question": "A class has 40 students, including 25 boys and 15 girls. ![image](question_images/a284h5iuh38.jpg) Find the percentage of girls in the class.",
    "sub_answer": "37.5%.",
    "sub_solution": "Percentage of girls = (15/40) * 100 = 37.5%."
  }
]
```
      Now, please split the following question according to the above requirements:
      [Question]
      {input_question}
      
      [Answer]
      {input_answer}
      
      [Solution]
      {input_solution}
      """
    
    def build_prompt(self, need_fields, **kwargs):
        # 校验缺失字段
        missing = [f for f in need_fields if f not in kwargs]
        if missing:
            if self.on_missing == "raise":
                raise KeyError(f"Missing fields for prompt: {missing}")
            # 宽松模式：用空串补齐
            for f in missing:
                kwargs[f] = ""
        prompt = self.f_str_template
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt
        
@PROMPT_REGISTRY.register()
class TypeClassifyPrompt(DIYPromptABC):
    def __init__(self, f_str_template: str = "{input_text}", on_missing: str = "raise"):
        self.f_str_template ='''
[Role]
You are an education expert familiar with textbook question formats at high school and university levels.
Your task is to determine the question type based on the question and answer provided.

[Possible Categories]
Choose exactly one of the following types:

1. Proof problem - requires proving a statement, identity, inequality, or property.

2. Explanation problem - asks for reasoning, causes, interpretation, principle, or conceptual explanation.

3. Fill-in problem - asks to fill in blanks, complete missing expressions, or supply intermediate steps.

4. Calculation problem - involves explicit numerical or symbolic computation, formula manipulation, or value derivation.
Even if the final answer is a short conclusion such as “thus xxx increases” or “so the velocity decreases,”
it should still be considered a Calculation problem if the majority of the reasoning is computational.

5. Multiple-choice problem - asks to choose or identify the correct option (e.g., “Which of the following…”).

6. Sketching/Plotting problem - requires sketching a figure, diagram, graph, or geometric representation.

7. Other - for tasks that don't fit any of the above types.

[Judgment Rules]

1. If the problem explicitly says “prove,” “show that,” “derive,” and does not have a short final answer → classify as Proof problem.

2. If it mainly contains explanations, reasoning, or conceptual analysis without detailed calculation → Explanation problem.

3. If the question has blanks, missing terms, or placeholders (e.g., “( )” or “____”), or the question seems **incomplete** → Fill-in problem.

4. If there are multiple formula derivations, substitutions, or numeric results → Calculation problem,
even if followed by a brief explanatory conclusion.

5. If it asks to select the correct answer among options (A/B/C/D, etc.) → Multiple-choice problem.

6. If the question explicitly requires producing a figure, diagram, plot, or geometric construction → Sketching/Plotting problem.

7. If none of these clearly apply or the problem type is mixed → Other.

[Output Format]
Return a JSON object with the following fields:
{
  "type": "Calculation | Proof | Explanation | Fill-in | Multiple-choice | Sketching/Plotting | Other",
  "reason": "Brief justification for the classification."
}

Please determine the type of the following question and output only one of the above category names.
(Proof, Explanation, Fill-in, Calculation, Multiple-choice, Sketching, Other).

[Question]
{input_question}

[Answer]
{input_answer}
        '''
        
    def build_prompt(self, need_fields, **kwargs):
        # 校验缺失字段
        missing = [f for f in need_fields if f not in kwargs]
        if missing:
            if self.on_missing == "raise":
                raise KeyError(f"Missing fields for prompt: {missing}")
            # 宽松模式：用空串补齐
            for f in missing:
                kwargs[f] = ""
        prompt = self.f_str_template
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt
    
@PROMPT_REGISTRY.register()
class QAFilterPrompt(DIYPromptABC):
    """
    用于过滤不合适的问答对的Prompt
    """
    def __init__(self):
        self.f_str_template = """
        [Role]
        You are an education expert familiar with textbook question formats at high school and university levels.
        Your task is to determine whether the provided question and answer pair is suitable to serve as a problem in an exam.
        
        Question: {input_question}
        
        Answer: {input_answer}
        
        [Criteria]
        1. Clarity: The question must be suitable for an exam setting, meaning it should raise **a clear problem** that requires a specific solution.
            Examples, **statements without questions**, open-ended discussions and other context that do not pose a clear problem are not suitable.
            Questions like "Give an example of..." that can have many valid answers are also not suitable.
            You should be particularly careful with questions that **only provide a topic or theme** without a specific problem to solve.
            For instance, "all primes less than 100" is not a valid question, because it does not specify what to do (listing, counting, ...) with those primes.
            Instead, a question like "List all primes less than 100" or "How many primes are there less than 100?" would be suitable.
        2. Relevance: The answer must directly address the question asked.
            If the answer seems to be addressing a different question and is wrongly paired with the given question, it is not suitable.
        3. Completeness and Self-Containment: The question and answer should be complete and self-contained, providing all necessary information for understanding and solving it without requiring external context.
           Questions that rely heavily on prior context or external references are not suitable.
           Answers such as "Refer to theorem X", "Corollary of previous result", "Answered in the text above", "Omitted for brevity" are not acceptable.
           Incomplete questions or answers that leave out critical information are also not suitable.
        4. Explicit Task Requirement: The question must contain an explicit task phrase (such as "compute", "determine", "find", "prove", "list", "show", "give the value of", etc.).
            Pure expressions or noun phrases are NOT acceptable even if they are commonly understood as implicit tasks in mathematical contexts. 
            If the question does not include an explicit verb specifying what the student must do, it must be judged unsuitable.
            Of course, if the question is in a multiple-choice or fill-in-the-blank format, the choices or blanks themselves will serve as the explicit task requirement.
           
        [Important Notice]
        1. You do not need to evaluate the correctness of the answer, only whether it is appropriate and complete in relation to the question.
        2. Short answer with no explanation (calculation, proof, counterexample, ...) is acceptable as long as it directly addresses the question.
        3. There might be figures in the question or answer, represented as `![image](image_url)`. However, we do not give you that.
            You can assume that if the question or answer contains such figure references, they are correctly placed and provide necessary information. 
        4. Sometimes in a fill-in question, the blanks like "___" may be missing due to OCR errors. In this case, if the question is otherwise clear and complete, you can still judge it as suitable.
        5. You should be very strict in your evaluation. If any of the criteria above are not fully met, the question-answer pair should be considered unsuitable.
           
        [Output Format]
        Return a JSON object with the following fields:
        {
        "reason": "Brief justification of your judgement."
        "judgement": "true | false",
        } 
              
        Your judgment:
        """
    
    def build_prompt(self, need_fields, **kwargs):
        # 校验缺失字段
        missing = [f for f in need_fields if f not in kwargs]
        if missing:
            if self.on_missing == "raise":
                raise KeyError(f"Missing fields for prompt: {missing}")
            # 宽松模式：用空串补齐
            for f in missing:
                kwargs[f] = ""
        prompt = self.f_str_template
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt