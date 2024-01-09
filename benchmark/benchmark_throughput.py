"""Benchmark offline inference throughput."""
# python3 benchmark_serving.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer ckpt/FlagAlpha/Llama2-Chinese-13b-Chat/
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizerBase
from cacheflow.master.simple_frontend import SimpleFrontend
from cacheflow.master.server import Server
from cacheflow.sampling_params import SamplingParams
from cacheflow.utils import get_gpu_memory, get_cpu_memory


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[List[int], int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    #prompt_token_ids = tokenizer(prompts).input_ids
    prompt_tokens = tokenizer(prompts, add_special_tokens=False)
    completions = [completion for _, completion in dataset]
    #completion_token_ids = tokenizer(completions).input_ids
    completion_tokens = tokenizer(completions)
    tokenized_dataset = []
    for i in range(len(dataset)):
        #output_len = len(completion_token_ids[i])
        output_len = len(completions[i])
        tokenized_dataset.append((prompts[i], output_len))
    # Filter out if the prompt length + output length is greater than 2048.
    tokenized_dataset = [
        (prompts, output_len)
        for prompts, output_len in tokenized_dataset
        if len(prompts) + output_len <= 2048
    ]

    # Sample the requests.
    sampled_requests = random.sample(tokenized_dataset, num_requests)
    return sampled_requests



def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Create a server.
    server = Server(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        gpu_memory=get_gpu_memory(),
        cpu_memory=get_cpu_memory(),
    )

    # Create a frontend.
    frontend = SimpleFrontend(
        model_name=args.model,
        block_size=args.block_size,
    )

    sampling_params_dict = {
        'n': args.n,
        'temperature': 0.0 if args.use_beam_search else 1.0,
        'top_p': 1.0,
        'use_beam_search': args.use_beam_search,
        'stop_token_ids': set(),
        'max_num_steps': args.output_len,
    }
    sampling_params = SamplingParams.from_dict(sampling_params_dict)

    requests = sample_requests(args.dataset, args.num_prompts)

    # Add the requests to the server.
    for prompt,_, output_len in requests:
        sampling_params = SamplingParams(
            n=args.n,
            temperature=0.0 if args.use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=args.use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        frontend.query(prompt, sampling_params)
    server.add_sequence_groups(frontend.get_inputs())
        # # FIXME(woosuk): Do not use internal method.
        # llm._add_request(
        #     prompt="",
        #     sampling_params=sampling_params,
        #     prompt_token_ids=prompt_token_ids,
        # )
    start = time.time()
    # FIXME(woosuk): Do use internal method.
    server.step()
    end = time.time()
    total_num_tokens = sum(
        len(prompt) + output_len
        for prompt, output_len in requests
    )
    print(f"Throughput: {total_num_tokens / (end - start):.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)