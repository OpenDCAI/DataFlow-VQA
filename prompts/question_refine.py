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
        However, the question may have some missing "___" to indicate blanks.
        You can use the provided answer to help determine where the blanks should be placed.
        Ensure that the modified question clearly indicates all the blanks using "___".
        
        Question: {input_question}
        
        Answer: {input_answer}
        
        Example:
        Original Question: The capital of France is  and the capital of British is ___.
        Answer: Paris; London
        Modified Question: The capital of France is ___ and the capital of British is ___.
        
        [Important Notice]
        1. If the original question already has some "___", do not remove them. Instead, add any missing "___" based on the answer.
            If the question is already complete with all necessary blanks, return it as is.
        2. Do not change any other part of the question except for adding the missing "___" !!!

        [Output Format]
        Only output the full modified question with blanks represented by "___". Do not include any additional explanations or text.
        For example, output: The capital of France is ___ and the capital of British is ___.
        
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