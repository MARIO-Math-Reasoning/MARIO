import unittest
import sys
sys.path.append(".")

from math_evaluation.core.evaluations import is_equiv


def _test_is_equiv(test_in, test_out, verbose=False):
    output = is_equiv(test_in, test_out, verbose=verbose)
    if not output:
        print(f"Test not passed: {test_in} == {test_out}")
    return output


def _test_is_not_equiv(test_in, test_out, verbose=False):
    output = is_equiv(test_in, test_out, verbose=verbose)
    if output:
        print(f"Test not passed: {test_in} != {test_out}")
    return output


class TestIsEquiv(unittest.TestCase):

    def test_fractions(self):
        test_in = "\\tfrac{1}{2} + \\frac1{72}"
        test_out = "\\\\frac{1}{2} + 2/3"
        self.assertFalse(_test_is_not_equiv(test_in, test_out, verbose=True))

    def test_order(self):
        test_in = "10, 4, -2"
        test_out = "4, 10, -2"
        self.assertFalse(_test_is_not_equiv(test_in, test_out, verbose=True))

    def test_order2(self):
        test_in = "10, 4, 2"
        test_out = "4, 12, 2"
        self.assertFalse(_test_is_not_equiv(test_in, test_out, verbose=True))

    def test_dfrac(self):
        test_in = "\\tfrac{1}{2} +\\! \\frac1{72}"
        test_out = "\\\\dfrac{1}{2} +\\frac{1}{72}"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

    def test_units(self):
        test_in = "10\\text{ units}"
        test_out = "10 "
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

        test_in = "58 square units"
        test_out = "58"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

    def test_units2(self):
        test_in = "10\\text{ units}"
        test_out = "100 "
        self.assertFalse(_test_is_not_equiv(test_in, test_out, verbose=True))

    def test_dollars(self):
        test_in = "10"
        test_out = "\\$10"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

    def test_parentheses(self):
        test_in = "\\left(x-2\\right)\\left(x+2\\right)"
        test_out = "(x-2)(x+2)"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

    def test_decimals(self):
        test_in = "0.1, 4, 2"
        test_out = "4, .1, 2"
        self.assertFalse(_test_is_not_equiv(test_in, test_out, verbose=True))

    def test_decimals2(self):
        test_in = "0.1"
        test_out = ".1"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

    def test_percentage(self):
        test_in = "10\\%"
        test_out = "10"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

    def test_sqrt(self):
        test_in = "10\\sqrt{3} + \\sqrt4"
        test_out = "10\\sqrt3 + \\sqrt{4}"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

    def test_frac(self):
        test_in = "\\frac34i"
        test_out = "\\frac{3}{4}i"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

    def test_tfrac(self):
        test_in = "\\tfrac83"
        test_out = "\\frac{8}{3}"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

    def test_expression(self):
        test_in = "5x - 7y + 11z + 4 = 0"
        test_out = "x + y - z + 2 = 0"
        self.assertFalse(_test_is_not_equiv(test_in, test_out, verbose=True))

        test_in = "11z^{17}(5+11z^{17})"
        test_out = "11z^17(11z^17+5)"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))

    def test_half(self):
        test_in = "1/2"
        test_out = "\\frac{1}{2}"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))
    
    def test_frac_sqrt(self):
        test_in = "$\frac{\sqrt{17}}{17}$"
        test_out = "$\\frac{\\sqrt{17}}{17}$"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))
    
    def test_frac_sqrt2(self):
        test_in = "$\frac{4\sqrt{5}}{5}$"
        test_out = "$\\frac{4\\sqrt{5}}{5}$"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))
    
    def test_frac2(self):
        test_in = "$1+3\\sqrt{2}$"
        test_out = "$1+3\\\\sqrt{2}$"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))
    
    def test_frac3(self):
        test_in = "$\\sqrt{5}$"
        test_out = "$\\\\sqrt{5}$"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))
    
    def test_half2(self):
        test_in = "1/2"
        test_out = "\\frac12"
        self.assertTrue(_test_is_equiv(test_in, test_out, verbose=True))


if __name__ == '__main__':
    unittest.main()
