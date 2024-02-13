"""
author: lmp-decaderan
email: ldecaderan@gmail.com
"""
import argparse

import ast
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional, Type, List

from pydantic import BaseModel, Field, root_validator
from timeout_decorator import timeout


TIMEOUT_SECONDS = 1500
TIMEOUT_MESSAGE = f"Execution of the code snippet has timed out for exceeding {TIMEOUT_SECONDS} seconds."


def extract_content(text):
    pattern = r'print\((.*?)\)'
    matches = re.findall(pattern, text)
    if len(matches) < 1:
        return ""
    return " ".join(matches)+":"


def __is_print_node(node: ast.AST) -> bool:
    
    if isinstance(node, ast.Expr) and \
        isinstance(node.value, ast.Call) and \
        isinstance(node.value.func, ast.Name) and \
        node.value.func.id == "print":
        return True
    elif isinstance(node, ast.If) or \
         isinstance(node, ast.While) or \
         isinstance(node, ast.For) or \
         isinstance(node, ast.FunctionDef):
        for sub_node in node.body:
            if __is_print_node(sub_node):
                return True
    return False


def find_print_node(body: List[ast.AST]) -> List[int]:
    """Find the python print node in the tree.body.

    Args:
        body (List[ast.AST]): The body of the AST

    Returns:
        List[int]: The index of the python print node
    """
    print_index = []
    for idx, node in enumerate(body):
        if __is_print_node(node):
            print_index.append(idx)
    return print_index


def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query


class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")
    

class PythonInterpreter(BaseModel):
    """A tool for running python code snippet."""

    name: str = "python_interpreter"
    description: str = (
        "A Python shell. Use this to execute python commands. "
    )
    description_zh: str = (
        "Python 交互式 shell。使用此工具来执行 Python 代码。"
    )
    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)
    sanitize_input: bool = True
    args_schema: Type[BaseModel] = PythonInputs
    use_signals: bool = True    # need to set False for async run

    @root_validator(pre=True)
    def validate_python_version(cls, values: Dict) -> Dict:
        """Validate valid python version."""
        if sys.version_info < (3, 9):
            raise ValueError(
                "This tool relies on Python 3.9 or higher "
                "(as it uses new functionality in the `ast` module, "
                f"you have Python version: {sys.version}"
            )
        return values

    def _base_run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        def _sub_run(bodys):
            io_buffer = StringIO()
            module = ast.Module(bodys[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(bodys[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore

            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, self.globals, self.locals)
                    if ret is None:
                        return True, io_buffer.getvalue()
                    else:
                        return True, ret
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return False, io_buffer.getvalue()
            
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
            tree = ast.parse(query)
            print_indexs = find_print_node(tree.body)
            if len(print_indexs) == 0:
                print_indexs = [len(tree.body) - 1]
            ret_strs = []
            if len(print_indexs) == 1:
                run_flag, ret = _sub_run(tree.body)
                #if not run_flag:
                #    return f"Error: {ret}"
                return f"{ret}"
            for start_idx, end_idx in zip([-1] + print_indexs, print_indexs):
                node_source = ast.get_source_segment(query, tree.body[end_idx])
                run_flag, ret = _sub_run(tree.body[start_idx + 1:end_idx + 1])
                #if not run_flag:
                #    ret_strs.append(f"{extract_content(node_source)} Error {ret}")
                #    break
                ret_strs.append(f"{extract_content(node_source)} {ret}")
            return "".join(ret_strs)
        except Exception as e:
            return "{}: {}".format(type(e).__name__, str(e))
    
    def run(
        self,
        query: str,
    ) -> str:

        @timeout(TIMEOUT_SECONDS, use_signals=self.use_signals, exception_message=TIMEOUT_MESSAGE)
        def base_run(query: str) -> str:
            return self._base_run(query)
        
        try:
            ret = base_run(query)
            return ret
        except Exception as e:
            return "{}: {}".format(type(e).__name__, str(e))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--testcase', type=str, default="```python\nprint(1)\nprint(2)\n```")
    # input args
    args = args.parse_args()
    return args


if __name__ == '__main__':
    # example usage
    args = parse_args()

    python = PythonInterpreter(globals=globals(), locals=None)
    #print(python.run("```print('unittest')```"))
    #print(python.run(args.testcase))
    #code_snippet = "```python\ntest_scores = [65, 94, 81, 86, 74]\naverage = float(sum(test_scores)) / len(test_scores)\nif average >= 80:\n    print(\"Wilson's current math grade is:\", average)\nelse:\n    print(\"Wilson needs to study more to achieve a grade of 80 or higher.\")\n```"
    code_snippet = "```python\ndef foo(x):\n    if x > 0:\n        print(\"x > 0\")\n    else:\n        print(\"x <= 0\")\n\nfoo(a)\n```"
    print(python.run(code_snippet))

