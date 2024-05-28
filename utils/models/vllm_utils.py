from typing import List

import torch
from vllm import LLM, SamplingParams


class VLLM:
    def __init__(self, name, tokenizer_name=None, num_gpus=1):
        dtype = "float16"
        if torch.cuda.is_bf16_supported():
            dtype = "bfloat16"

        self.name = name

        max_model_len = None

        self.model = LLM(
            model=self.name,
            tokenizer=tokenizer_name,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,
        )

    def completions(
        self,
        prompts: List[str],
        use_tqdm=False,
        **kwargs,
    ) -> List[List[str]]:
        prompts = [prompt.strip() for prompt in prompts]
        params = SamplingParams(**kwargs)
        request_outputs = self.model.generate(prompts, params, use_tqdm=use_tqdm)
        outputs = []
        for request_output in request_outputs:
            completion_texts = []
            for completion_output in request_output.outputs:
                completion_texts.append(completion_output.text.strip())
            outputs.append(completion_texts)
        return outputs
