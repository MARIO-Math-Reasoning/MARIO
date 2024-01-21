
from math_evaluation.core.constants import *
from math_evaluation.core.latex_normalize import string_normalize
from math_evaluation.core.latex_normalize import ( split_tuple, 
                                _is_frac, 
                                _str_is_int)
from math_evaluation.core.latex_parser import are_equal_under_sympy


def is_equiv(str1: str, str2: str, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    str_pass, str1_normalized, str2_normalized = is_normalized_string_equival(str1, str2, verbose=verbose)
    if str_pass:
        return True
    
    try:
        latex_pass = is_latex_equiv(str1_normalized, str2_normalized, verbose=verbose)
        return latex_pass
    except:
        return False


def is_normalized_string_equival(str1: str, str2: str, verbose=False):
    try:
        ss1 = string_normalize(str1)
        ss2 = string_normalize(str2)
        if verbose:
            print(ss1, ss2)
        if ss1 == ss2:
            return True, ss1, ss2
        return ss1 == ss2, ss1, ss2
    except:
        return str1 == str2, str1, str2


def is_latex_equiv(ground_truth_normalized: str, given_normalized: str, verbose=False):

    if len(given_normalized) == 0:
        return False

    is_correct = are_equal_under_sympy(ground_truth_normalized, given_normalized)
    if is_correct:
        return True
    
    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if verbose:
        print(ground_truth_elems, given_elems)

    is_correct = False
    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct