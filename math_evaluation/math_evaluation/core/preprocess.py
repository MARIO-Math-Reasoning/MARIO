import re
from bs4 import BeautifulSoup, NavigableString

def default_preprocess(data):
    return data


def lastline(data):
    return data.split("\n")[-1]


def gsm8k_label(data):
    return  data.split("####")[-1].strip()


def gsm8k_predict(data):
    data = lastline(data)
    extracted_number_list = re.findall(r"(\-?[0-9\.\,]+)", data)
    return  extracted_number_list[0]


# def boxed_predict(data):
#     first_pattern = r"boxed\{(.*?)\}$"
#     pattern = r"boxed\{(.*?)\}"
#     match = re.search(first_pattern, data)
#     if match is None:
#         match = re.search(pattern, data)

#     if match is None:
#         return ""
#     return match.group(1)

def boxed_predict(data):
    pattern_list = ["\\boxed", "\\fbox"]
    begin_index = None
    for pattern in pattern_list:
        index = data.rfind(pattern)
        if index != -1:
            begin_index = index
            break
    if begin_index is None:
        return None

    i = begin_index
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(data):
        if data[i] == "{":
            if num_left_braces_open == 0:
                left_brace_idx = i
            num_left_braces_open += 1
        if data[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    answer = data[left_brace_idx+1:right_brace_idx]
    return answer


def metamath_predict(data):
    """
    从给定字符串中提取答案标签

    参数：
    data：给定的字符串

    返回值：
    提取出的标签字符串
    """

    return data.split("The answer is:")[-1].strip()


def mathsteps_predict(data):
    """
    从给定字符串中提取答案标签

    参数：
    data：给定的字符串

    返回值：
    提取出的标签字符串
    """
    data_split = data.split("Final Answer:")
    if len(data_split) < 2:
        return ""
    return data_split[-1].strip()


def react_zh_predict(data):
    """
    从给定字符串中提取答案标签

    参数：
    data：给定的字符串

    返回值：
    提取出的标签字符串
    """
    if data.startswith("‘"):
        data = data[1:]
    if data.endswith("’"):
        data = data[:-1]
    if "答案是" in data:
        data_split = data.split("答案是")
    else:
        data_split = data.split("答案：")
    if len(data_split) < 2:
        return ""
    return data_split[-1].strip()


def __html_to_text(soup):
    default_tag = "{}"
    tag2ltx = {
        "sup": "^{}", 
        "sub": "_{}",
        "p": "{} ",
        "div": "{}",
        "u": "\\underline{{{}}}",
        "b": "\\textbf{{{}}}",
        "i": "\\textit{{{}}}",
        "em": "\\emph{{{}}}",
        "span": "\\text{{{}}}",
        "br": "\\\\",
        }
    res_str = ""
    for item in list(soup.children):
        if isinstance(item, NavigableString):
            res_str += tag2ltx.get(item.name, default_tag).format(item.text)
        else:
            res_str += __html_to_text(item)
    return res_str


def html_predict(data):
    """
    从给定html字符串中提取答案

    参数：
    data：给定的字符串

    example:
    >>> html_predict("<div>123</div>")
    "123"
    >>> html_predict("<p><u>佩玲</u>的日薪较<u>志伟</u>高$$20\\%$$而<u>志伟</u>的日薪较<u>洁仪</u>低$$20\\%$$． 已知<u>志伟</u>的日薪为＄$$480$$．</p>")
    "佩玲的日薪较志伟高$$20\\%$$而志伟的日薪较洁仪低$$20\\%$$． 已知志伟的日薪为＄$$480$$．"

    返回值：
    提取出的标签字符串
    """
    soup = BeautifulSoup(data,'html.parser')
    res_str = __html_to_text(soup)
    return res_str