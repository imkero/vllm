from vllm import SamplingParams
from transformers import AutoTokenizer
from vllm_server import VllmServer


def expected_usage():
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    server = VllmServer(model=model_path, enforce_eager=True)
    server.launch()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    req1 = server.add_request(
        prompt={
            "prompt":
            tokenizer.apply_chat_template(
                conversation=[{
                    "role": "user",
                    "content": "Hello!"
                }],
                add_generation_prompt=True,
                tokenize=False,
            ),
        },
        sampling_params=SamplingParams(temperature=0),
    )

    req2 = server.add_request(
        prompt={
            "prompt":
            tokenizer.apply_chat_template(
                conversation=[{
                    "role": "user",
                    "content": "What is the capital of France?"
                }],
                add_generation_prompt=True,
                tokenize=False,
            ),
        },
        sampling_params=SamplingParams(temperature=0),
    )

    print(req1.result())
    print(req2.result())

    server.shutdown()


if __name__ == "__main__":
    expected_usage()
