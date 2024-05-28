import json
import os

import openai
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def pricing_info(model):
    if model.startswith("gpt-4") and model.endswith("preview"):
        input_rate = 0.01
        output_rate = 0.03
    elif model == "gpt-4":
        input_rate = 0.03
        output_rate = 0.06
    elif model == "gpt-4-32k":
        input_rate = 0.06
        output_rate = 0.12
    elif model == "gpt-3.5-turbo-0125":
        input_rate = 0.0005
        output_rate = 0.0015
    elif model == "gpt-3.5-turbo-instruct":
        input_rate = 0.0015
        output_rate = 0.0020
    else:
        raise ValueError(f"Model {model} not supported.")
    return input_rate, output_rate


class OpenAILLM:
    def __init__(
        self,
        name,
    ):
        self.name = name
        self.batch_size = 100
        self.requests_per_minute = 100
        self.limiter = AsyncLimiter(self.requests_per_minute, 60)
        self.client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    async def get_completion_text_async(self, messages, **kwargs):
        async with self.limiter:
            try:
                # Assuming you have a session and client setup for OpenAI
                completion = await self.client.chat.completions.create(
                    model=self.name, messages=messages, **kwargs
                )
                content = completion.choices[0].message.content.strip()
                if kwargs.get("response_format") == {"type": "json_object"}:
                    content = json.loads(content)
                # usage = completion.usage
                return content  # , usage
            except openai.APIConnectionError as e:
                print("APIConnectionError: The server could not be reached")
                print(
                    e.__cause__
                )  # an underlying Exception, likely raised within httpx.
            except openai.RateLimitError as e:
                print(
                    "RateLimitError: A 429 status code was received; we should back off a bit."
                )
            except openai.APIStatusError as e:
                print("APIStatusError: Another non-200-range status code was received")
                print(e.status_code)
                print(e.response)
            except Exception as e:
                print(f"Error during OpenAI API call: {e}")
                return ""  # , {}

    async def completions(
        self,
        messages,
        **kwargs,
    ):
        assert isinstance(messages, list)
        assert list(messages[0][0].keys()) == ["role", "content"]

        result_responses = []

        for start_idx in tqdm(
            range(0, len(messages), self.batch_size), desc="Processing batches"
        ):
            end_idx = start_idx + self.batch_size

            batch_prompts = messages[start_idx:end_idx]

            batch_responses = await tqdm_asyncio.gather(
                *[
                    self.get_completion_text_async(prompt, **kwargs)
                    for prompt in batch_prompts
                ]
            )

            result_responses.extend(batch_responses)

        return result_responses


if __name__ == "__main__":
    print("Hello, World!")

    model = OpenAILLM("gpt-3.5-turbo")

    responses = model.completions(
        model="gpt-3.5-turbo",
        messages=[
            [{"role": "user", "content": "good morning? "}],
            [{"role": "user", "content": "what's the time? "}],
        ],
    )
    import pdb

    pdb.set_trace()
