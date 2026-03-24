from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

@PROMPT_REGISTRY.register()
class AddMissingBlankPrompt(PromptABC):
    """
    用于补全填空题中的横线
    """
    def __init__(self):
        self.f_str_template = """
        [Role]
        You are an education expert familiar with textbook question formats at high school and university levels.
        You will be given a "fill-in-the-blank" question along with its answer.
        However, the question may have some missing placeholders to indicate blanks.
        You can use the provided answer to help determine where the blanks should be placed.
        Ensure that the modified question clearly indicates all the blanks using placeholders.
        
        Question: {input_question}
        
        Answer: {input_answer}
        
        [Important Notice]
        1. If the original question already has some placeholders (coule be in different forms such as "()", "__", "____"), do not remove them. Instead, add any missing "___" based on the answer.
            If the question is already complete with all necessary blanks (regardless of the form of placeholders), return "ORIGINAL" (no quote).
        2. Do not change any other part of the question except for adding the missing "___" !!!
        
        [Examples]
        Original Question: The capital of France is  and the capital of British is ____.
        Answer: Paris; London
        Return: The capital of France is ___ and the capital of British is ____.
        
        Original Question: The two right sides of a triangle are 3 and 4, then the third side is.
        Answer: 5
        Return: The two right sides of a triangle are 3 and 4, then the third side is ___.
        
        Original Question: The area of a circle with radius r is ( ).
        Answer: πr^2
        Return: ORIGINAL

        [Output Format]
        Only output the full modified question with blanks represented by "___". Do not include any additional explanations or text.
        
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
