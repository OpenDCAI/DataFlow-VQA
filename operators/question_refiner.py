import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class AddMissingBlankOperator(OperatorABC):
    def __init__(
            self,
            llm_serving: LLMServingABC, 
            prompt_template,
        ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template
        if prompt_template is None:
            raise ValueError("prompt_template cannot be None")


    def run(
            self, 
            storage: DataFlowStorage,
            output_key: str = "question",
            **input_keys
        ):
        self.storage: DataFlowStorage = storage
        self.output_key = output_key
        self.logger.info("Running AddMissingBlankOperator...")
        self.input_keys = input_keys

        need_fields = set(input_keys.keys())

    # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")
        llm_inputs = []


        # Only process rows where type == "fill-in"
        if 'type' not in dataframe.columns:
            self.logger.warning("No 'type' column found, skipping LLM generation.")
            generated_outputs = []
        else:
            mask = dataframe['type'] == "Fill-in"
            indices = dataframe.index[mask].tolist()
            if not indices:
                self.logger.info("No rows with type=='Fill-in' to process.")
                generated_outputs = []
            else:
                for idx in indices:
                    row = dataframe.loc[idx]
                    key_dict = {key: row[input_keys[key]] for key in need_fields}
                    prompt_text = self.prompt_template.build_prompt(need_fields, **key_dict)
                    llm_inputs.append(prompt_text)
                self.logger.info(f"Prepared {len(llm_inputs)} prompts for LLM generation.")
                generated_outputs = self.llm_serving.generate_from_input(llm_inputs)
            # write generated outputs back only to the selected rows (preserve other rows as None)
            for idx, gen_output in zip(indices, generated_outputs):
                if gen_output != "ORIGINAL":
                    dataframe.at[idx, output_key] = gen_output

        output_file = self.storage.write(dataframe)
        return output_key