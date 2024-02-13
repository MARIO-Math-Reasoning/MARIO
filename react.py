"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import re
import argparse
from typing import List, Dict
from termcolor import colored

from vllm.outputs import RequestOutput
from timeout_decorator import timeout

from python_tool import PythonInterpreter


SFT_PROMPT = "Question: {question}\n\nSolution:\n\n{partial_solution}"
PRIMER = "<p>\n"
STOP = ["\n</code>", "</code>"]
CODE_LTAG = "<code>"
CODE_RTAG = "</code>"


def _python_ast_init():
    python = PythonInterpreter(globals=globals(), locals=None)
    return python


def tool_wrapper(tool):
    def _tool(query):
        return tool.run(query)
    return _tool


def no_action_wrapper(tool):
    def _tool(query):
        return "No action, no observation. Please continue to solve."
    return _tool


# We define a dummy tool, to implement multiple tools.
tools = {
    "None": no_action_wrapper(_python_ast_init()),
    "python_interpreter": tool_wrapper(_python_ast_init()),
}


def action_execution(parser_results: List[Dict[str, str]]) -> str:

    @timeout(30)
    def _action_execution(parser_results: List[Dict[str, str]]) -> str:
        cur_action = parser_results[-1]["action"]
        tool_func = tools[cur_action]

        # first, execute historical action inputs with the same action, but not output
        for history_act in parser_results[:-1]:
            if history_act["action"] == cur_action:
                _ = tool_func(history_act["action_input"])
        
        # then, execute current action input, and return output
        observation = str(tool_func(parser_results[-1]["action_input"]))
        del tool_func
        return observation 
    
    try:
        observation = _action_execution(parser_results)
    except Exception as e:
        observation = "{}: {}".format(type(e).__name__, str(e))

    return observation


class STEP(object):
    
    def __init__(self,
        text: str = "",
        action: str = "",
        action_input: str = "",
        final_answer: str = "",
        depth: int = 0,
    ):
        self.text = text
        self.action = action
        self.action_input = action_input
        self.final_answer = final_answer
        self.depth = depth

        self.next_step = None
        self.is_terminal = False


class ReactSolver(object):

    def __init__(self, args, question: str):
        self.args = args
        self.question = question

        self.start_step = STEP()

        self.current_step = self.start_step
        self.step_texts = []
        self.step_actions = []

    def step_generate_flag(self) -> bool:
        return not self.current_step.is_terminal and self.current_step.depth <= self.args.max_depth
    
    def get_llm_request(self) -> str:
        # get partial solution
        _partial_solution = "\n\n".join(self.step_texts)
        if _partial_solution:
            partial_solution = f"{_partial_solution}\n\n{PRIMER}"
        else:
            partial_solution = f"{PRIMER}"
        
        # create prompt
        prompt = SFT_PROMPT.format(question=self.question, partial_solution=partial_solution)
        return prompt
    
    def step_generate(self, output: RequestOutput) -> None:
        """process output from vllm
        e.g.,

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        sampled_step_result = (PRIMER + output.outputs[0].text).strip()
        # parsing code snippet
        step_result, parser_result = self.action_parser(sampled_step_result)
        self.process_step_result(step_result, parser_result, "Output")

    def process_step_result(
        self, 
        step_result: str,
        parser_result: Dict[str, str],
        observation_key: str,
    ) -> None:
        if self.args.verbose:
            print(colored(f"{step_result}", "green"))

        # create new step
        new_step = STEP()
        new_step.depth = self.current_step.depth + 1

        if parser_result is None:
            new_step.is_terminal = True
            new_step.text = step_result
            new_step.final_anser = "Cannot generate parsable text."
        elif parser_result["final_answer"]:
            new_step.is_terminal = True
            new_step.text = step_result
            new_step.final_answer = parser_result["final_answer"]
        elif parser_result["action"]:
            new_step.action = parser_result["action"]
            new_step.action_input = parser_result["action_input"]
            # update step_actions
            self.step_actions.append(parser_result)

            # get observation
            observation = action_execution(self.step_actions)

            if self.args.verbose:
                print(colored(f"{observation_key}: {observation}\n", "yellow"))
            
            new_step.text = f"{step_result}\n{observation_key}: {observation}"
        else:
            print("WARNING:")
            new_step.text = step_result
        
        # update step_texts
        self.step_texts.append(new_step.text)
        # update current step
        self.current_step.next_step = new_step
        self.current_step = new_step

    def action_parser(self, text: str):
        includes_answer = "Final Answer:" in text
        regex = r"{code_ltag}[\s]*(.*)".format(code_ltag=CODE_LTAG)
        code_match = re.search(regex, text, re.DOTALL)

        parser_result = { 
            "action": "",
            "action_input": "",
            "final_answer": "",
        }

        if code_match:
            if includes_answer:
                print(f"Warning: Incorrect format generated: `{text}`")
                return text, None
            
            text = f"{text}\n{CODE_RTAG}"
            code_snippet = code_match.group(1)
            parser_result["action"] = "python_interpreter"
            parser_result["action_input"] = code_snippet.strip(" ").strip('"')
            return text, parser_result

        elif includes_answer:
            parser_result["final_answer"] = text.split("Final Answer:")[-1].strip()
            return text, parser_result
        
        else:
            print(f"Warning: Could not parse LLM output: `{text}`")
            return text, None


def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument('-c', '--checkpoint_dir', type=str, required=True, help="folder of model checkpoint.")
    args.add_argument('--max_depth', type=int, default=8, help="maximum step of solution")
    args.add_argument('--verbose', action="store_true", help="print intermediate result on screen")
    args.add_argument('--temperature', type=float, default=0, help="for sampling")

    args.add_argument('-q', '--question', type=str, default=None, help="question")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    # the following script shows an example to solve one single question.
    from vllm import LLM, SamplingParams

    args = parse_args()

    # init llm
    llm = LLM(model=args.checkpoint_dir, tensor_parallel_size=1, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=-1,
        top_p=1.0,
        use_beam_search=False,
        best_of=1,
        max_tokens=2048, 
        n=1, 
        stop=STOP,
    )

    # define question and solver
    if args.question:
        question = args.question
    else:
        # an example question
        question = "Given complex number $(a+i)(1-ai)=2,\;a \in \mathbb{R}$, find $a$."
    if args.verbose:
        print(colored(f"Question: {question}\n", "red"))
    
    solver = ReactSolver(args, question)

    # run solver
    while solver.step_generate_flag():
        prompt = solver.get_llm_request()
        prompts = [prompt]
        outputs = llm.generate(prompts, sampling_params)
        solver.step_generate(outputs[0])
    
    # save solution
    full_solution = "\n\n".join(solver.step_texts)
    with open("log.txt", "w") as f:
        f.write(full_solution)
