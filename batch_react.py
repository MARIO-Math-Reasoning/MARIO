"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import json
import argparse
from termcolor import colored
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from react import ReactSolver, STOP


TIMEOUT_SECONDS = 40


def batch(iterable, n=-1):
    l = len(iterable)
    if n < 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def generator(solver, output):
    solver.step_generate(output)
    return solver


def batch_main(args, llm, sampling_params, questions, max_depth):

    try:
        solvers = [ReactSolver(args, question) for question in questions]

        for step in tqdm(range(max_depth), desc="Step Processing"):
            prompts = []
            stop_words = []
            epoch_solvers = []
            next_solvers = []
            for solver in solvers:
                if solver.step_generate_flag():
                    prompt_text = solver.get_llm_request()
                    prompts.append(prompt_text)
                    stop_words.extend(STOP)
                    epoch_solvers.append(solver)
                else:
                    next_solvers.append(solver)
            next_solver_span = len(next_solvers)
            if len(epoch_solvers) < 1:
                break
            sampling_params.stop = list(set(stop_words))
            outputs = llm.generate(prompts, sampling_params)

            with ProcessPool(max_workers=min(len(epoch_solvers), os.cpu_count())) as pool:
                future = pool.map(generator, epoch_solvers, outputs, timeout=TIMEOUT_SECONDS)
                iterator = future.result()

            if len(epoch_solvers) > 100:  
                progress_bar = tqdm(total=len(epoch_solvers), desc="Execute")  
            else:  
                progress_bar = None 

            while True:
                try:
                    result = next(iterator)
                    next_solvers.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    next_solvers.append(None)
                    print(error)
                except Exception as error:
                    print(error)
                    next_solvers.append(None)
                if progress_bar is not None:
                    progress_bar.update(1) 
            
            if progress_bar is not None:
                progress_bar.close() 
            
            # update solvers
            assert len(epoch_solvers) == len(next_solvers[next_solver_span:]), f"Data is not matched, {len(epoch_solvers)} vs {len(next_solvers[next_solver_span:])}."
            for idx, (ori_solver, new_solver) in enumerate(zip(epoch_solvers, next_solvers[next_solver_span:])):
                if new_solver is None:
                    next_solvers[next_solver_span + idx] = ori_solver
            solvers = next_solvers

    except Exception as e:
        print(colored(f"Exception: {e}", "red"))
        return [""] * len(questions)

    jsonlines = {}
    for solver in solvers:            
        try:
            solution = "\n\n".join(solver.step_texts)
            jsonlines[solver.question] = json.dumps(solution, ensure_ascii=False)
        except:
            jsonlines[solver.question] = json.dumps(solution, ensure_ascii=False)
    return jsonlines


def main(args):
    # init llm
    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
    llm = LLM(
        model=args.checkpoint_dir, 
        tensor_parallel_size=len(available_gpus), 
        trust_remote_code=False, 
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        top_k=args.top_k,
        best_of=args.best_of,
        use_beam_search=args.use_beam_search,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_generate_sample,
    )

    # load question file
    data = []
    with open(args.question_file, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # write results
    with open(f'{args.question_file}.prediction.jsonl', 'w') as writer:
        for cur_data in tqdm(batch(data, args.num_per_inference), desc="Main Processing"):
            questions = [d[args.question_key] for d in cur_data]
            try:
                jsonlines = batch_main(args, llm, sampling_params, questions, args.max_depth)
                for d in cur_data:
                    question = d[args.question_key]
                    d["react"] = jsonlines[question]
                    writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                    writer.flush()
            except Exception as e:
                print(colored(f"Batch Process Exception: {e}", "red"))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-q', '--question_file', type=str, required=True, help="file path of question file, should be jsonl format.")
    args.add_argument('--question_key', type=str, default="question", help="questioin key in json")
    args.add_argument('--num_per_inference', type=int, default=-1, help="number of questions per inference")
    
    # react
    args.add_argument('-c', '--checkpoint_dir', type=str, required=True, help="folder of model checkpoint.")
    args.add_argument('--max_depth', type=int, default=8, help="maximum step of solution")
    args.add_argument('--verbose', action="store_true", help="print intermediate result on screen")

    # vll
    args.add_argument('--max_tokens', type=int, default=2048, help="decoding tokens")
    args.add_argument('--temperature', type=float, default=0, help="for sampling")
    args.add_argument('--top_k', type=int, default=-1, help="for sampling")
    args.add_argument('--top_p', type=float, default=1.0, help="for sampling")
    args.add_argument('--use_beam_search', action="store_true", help="use beam search")
    args.add_argument('--best_of', type=int, default=1, help="for beam search")
    args.add_argument('--n_generate_sample', type=int, default=1, help="number of generated samples")
    args.add_argument('--seed', type=int, default=1234, help="random seed.")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
