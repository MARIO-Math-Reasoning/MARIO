"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import re
import argparse
import backoff 
import requests
from typing import Optional, List, Dict, Union
from termcolor import colored
from functools import partial

from prompts import (
    custom_prefix,
    custom_suffix,
    gsm8k_examples,
    math_examples,
)
from react import ReactSolver
from python_tool import PythonInterpreter


PRIMER = "Thought: "
STOP = ["\nObservation:", "Observation:"]


import openai

if openai.__version__ >= "1.0.0":
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60)
    def completions_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)

else:
    openai.api_key = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60)
    def completions_with_backoff(**kwargs):
        return openai.Completion.create(**kwargs)


def gpt(
    prompt: str, 
    model: str, 
    temperature: float = 0, 
    max_tokens: int = 4096, 
    n: int = 1, 
    stop: Union[Optional[str], List[str]] = None, 
    seed: int = None,
) -> List[str]:
    messages = [{"role": "user", "content": prompt}]
    response = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop, seed=seed)
    return [choice["message"]["content"] for choice in response["choices"]]


def create_prompt(
    question: str,
    partial_solution: str,
    dataset: str,
) -> str:
    if partial_solution:
        input_str = f"{question}\n\n{partial_solution}\n\n{PRIMER}"
    else:
        input_str = f"{question}\n\n{PRIMER}"
    
    if dataset == "gsm8k":
        custom_examples = gsm8k_examples
    elif dataset == "math":
        custom_examples = math_examples
    
    if len(custom_examples) > 1:
        example_prefix = "The following are %d demonstration examples." % len(custom_examples)
    elif len(custom_examples) == 1:
        example_prefix = "The following is a demonstration example."

    format_instructions = custom_prefix.format(
        tool_desc=f"{PythonInterpreter().name}: {PythonInterpreter().description}",
        tool_names=PythonInterpreter().name,
    )

    prompt_template = "\n\n".join([format_instructions, example_prefix, *custom_examples, custom_suffix])
    return prompt_template.format(input=input_str)


class GPTReactSolver(ReactSolver):

    def __init__(self, args, question: str):
        super().__init__(args, question)

        self.dataset = args.dataset
        self.llm = partial(gpt, model=args.gpt_model_id, temperature=args.temperature, seed=args.seed)
        self.stop = STOP
    
    def generate(self) -> str:
        while self.step_generate_flag():
            self.step_generate()
        return "\n\n".join(self.step_texts)
    
    def step_generate(self) -> None:
        partial_solution = "\n\n".join(self.step_texts)
        # get llm sample
        step_result, parser_result = self.get_parsable_samples(partial_solution)
        self.process_step_result(step_result, parser_result, "Observation")
    
    def get_parsable_samples(
        self,
        partial_solution: str, 
    ) -> Tuple[str, Dict[str, str]]:
        sampled_step_results = self.get_llm_samples(partial_solution, 1)

        try:
            assert len(sampled_step_results) == 1
            return self.action_parser(sampled_step_results[0])
        except Exception as e:
            n_samples = 3
            max_retry = 5 
            temperature = 0.7
            print(f"Exception: {e}. We will retry {max_retry} times by setting a larger temperature {temperature}, and generating {n_samples} samples.")
            retry_cnt = 0
            while retry_cnt < max_retry:
                sampled_step_results = self.get_llm_samples(partial_solution, n_samples, temperature)
                for step_result in sampled_step_results:
                    try:
                        return self.action_parser(step_result)
                    except Exception as e:
                        retry_cnt += 1
                        print(f"Exception: {e}. Retry {retry_cnt} failed.")
                        continue
            return step_result, None
    
    def get_llm_samples(
        self,
        partial_solution: str, 
        n: int = 1,
        temperature: float = 0,
    ) -> List[str]:
        # create prompt
        prompt = create_prompt(self.question, partial_solution, self.dataset)
        
        # get samples
        if temperature is None: # default llm
            samples = self.llm(prompt, n=n, stop=self.stop)
        else:
            diverse_llm = partial(gpt, model=self.args.gpt_model_id, temperature=temperature)
            samples = diverse_llm(prompt, n=n, stop=self.stop)
        
        # add primer key
        return [(PRIMER + " " + sample).strip() for sample in samples]
    
    def action_parser(self, text: str):
        includes_answer = "Final Answer:" in text
        regex = r"Action:[\s]*(.*?)[\s]*Action Input:[\s]*(.*)"
        action_match = re.search(regex, text, re.DOTALL)

        parser_result = {
            "action": "",
            "action_input": "",
            "final_answer": "",
        }

        if action_match:
            if includes_answer:
                print(f"Warning: Incorrect format generated: `{text}`")
                return text, None
            
            action = action_match.group(1).strip()
            if action not in [PythonInterpreter().name, "None"]:
                print(f"Warning: Incorrect format generated: `{text}`")
                return text, None

            action_input = action_match.group(2)
            tool_input = action_input.strip(" ").strip('"')

            parser_result["action"] = action
            parser_result["action_input"] = tool_input
            return text, parser_result

        elif includes_answer:
            parser_result["final_answer"] = text.split("Final Answer:")[-1].strip()
            return text, parser_result
        
        else:
            print(f"Warning: Could not parse LLM output: `{text}`")
            return text, None


def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument(
        '-g', '--gpt_model_id', 
        type=str, 
        default="gpt-3.5-turbo-1106",
        choices=["gpt-3.5-turbo-1106", "gpt-4-1106-preview"], 
        help="gpt model id",
    )
    args.add_argument('--max_depth', type=int, default=8, help="maximum step of solution")
    args.add_argument('--verbose', action="store_true", help="print intermediate result on screen")
    args.add_argument('--temperature', type=float, default=0, help="for sampling")
    args.add_argument('--seed', type=int, default=1234, help="for sampling")

    args.add_argument('--dataset', type=str, default="gsm8k", choices=["gsm8k", "math"], help="which prompt to use")
    args.add_argument('-q', '--question', type=str, default=None, help="question")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.verbose:
        print(colored(f"Question: {args.question}\n", "red"))
    
    solver = GPTReactSolver(args, args.question)
    solution = solver.generate()
