from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
'''
A collection of prompts for the general reasoning operator.
'''

@PROMPT_REGISTRY.register()
class GeneralAnswerWithSolutionGeneratorPrompt(PromptABC):
    '''
    The prompt for the answer generator based on the reference solution.
    '''
    def __init__(self):
        pass

    def build_prompt(self, question: str) -> str:
        """
        for general reasoning answer generation
        """
        prompt = (
            r'''You are an intelligent chatbot designed for producing the answer to the given reasoning task.
        Remember: DO NOT output anything else, only output the answer you generate.
        Generate a solution to the given task strictly following this format:
        1. Identify key components and premises of the task
        2. Apply relevant principles, theorems, or methods with step-by-step derivation or argument
        3. Perform any necessary calculations or logical checks with intermediate verification
        4. Present the final answer or conclusion in a clear, unambiguous notation

        You will be given a reasoning question (with images) and its reference solution. Use the reference solution to guide your answer generation.
        - If the reference solution gives a step-by-step derivation, follow the same steps in your answer.
        - If the reference solution only gives the final answer, derive the answer step-by-step in your response.
        - You can ignore the image information in the solution (not the question!), and focus on the textual content.
        - Don't just copy the reference solution; adapt it according to the format requirements above.
        - Don't skip necessary intermediate steps.
        
        Format Requirements:
        - Prefix each step with "→" (use the actual arrow symbol, not its Unicode escape sequence)
        - Ensure all symbols and special characters are presented using appropriate markup (e.g., LaTeX commands for mathematical symbols, code formatting for code snippets)

        Example Template:
        Question: Analyze the time complexity of the merge sort algorithm and prove its correctness.
        Reference solution: The algorithm uses divide-and-conquer and merging steps. The recurrence relation is T(n) = 2T(n/2) + O(n). By the Master Theorem, T(n) = O(n log n). 
        The base case and inductive step confirm correctness. Therefore, the algorithm runs in O(n log n) time and correctly sorts any input list.

        Solution:
        1. Identify components:
        → Algorithm uses divide-and-conquer to split the list in half
        → Merging step compares elements pairwise

        2. Apply principles:
        → Recurrence: T(n) = 2T(n/2) + O(n)
        → By Master Theorem, T(n) = O(n log n)

        3. Verification:
        → Check base case T(1) = O(1)
        → Inductive step holds for n = 2^k

        4. Conclusion:
        → The algorithm runs in \\boxed{O(n\\log n)} time and correctly sorts any input list.

        Here is the given task you need to solve:
        '''
        )
        return prompt + question + r'''Your response must start directly with "Solution:" without any preamble. Finish your response immediately after the solution.'''
