custom_prefix = """You are a powerful agent with broad math knowledge and great Python programming skills, but not good at accurate calculation on math equations. 
Answer the math questions as best you can. You have access to the following tool:

{tool_desc}

!!!Remember:
1. When solving math problem, you should think step by step, where each step includes 4 mini-steps Thought/Action/Action Input/Observation. 
2. If some step requires accurate calculation (including but not limited to numerical calculation and symbolic deduction), you should write Python code and execute for accurate result. 
3. The most related Python packages include `math`, `sympy`, `scipy` and `numpy`. 
4. Please use the following template.

Question: the input question

Thought: the text analysis, and list the math equations if necessary

Action: the action to take, should be `{tool_names}`, or None

Action Input: the Python Code in markdown format (if Action is None, it is None), e.g.,
```python
import math
theta = math.pi / 4
some_var = math.cos(theta)
print(some_var)
```

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: the final analysis of the answer

Final Answer: the concise answer without verbose context"""


custom_suffix = """Now! It's your turn.

Question: {input}"""


gsm8k_examples = [
    # example 1
    """Question: If $|x - 1| < 2$, then the maximum value of $f(x) = x^2 - 0.5x + 1.5$ is ()?

Thought: The maximum value of a function can be found by either finding the derivative and setting it equal to zero to find critical points, or by evaluating the function at the boundary points of the given interval. 
Since the function is a quadratic, it either opens upwards or downwards. 
If it opens upwards, the maximum value will be at one of the boundaries of the interval. 
If it opens downwards, the maximum will be at the vertex of the parabola. 
Because the coefficient of the $x^2$ term is 1, which means the parabola opens upwards. Therefore, the maximum value of the function will be at one of the boundaries of the interval defined by inequality $|x - 1| \\leq 2$. 

Action: python_interpreter

Action Input: ```python
from sympy import symbols, Abs, solve_univariate_inequality

# Define the variable
x = symbols('x')

# Solve the inequality
solution = solve_univariate_inequality(Abs(x-1)<=2, x)

print(solution)
```

Observation: (-1 <= x) & (x <= 3)

Thought: This implies $-1 \\leq x \\leq 3$. So, the maximum value of $f(x)$ will be either f(-1) or f(3).

Action: python_interpreter

Action Input: ```python
# Calculate each value
f_minus_1 = (-1)**2 - 0.5*(-1) + 1.5
f_3 = 3**2 - 0.5*3 + 1.5

# Compare which is larger
max_value = max(f_minus_1, f_3)

print(max_value)
```

Observation: 9.0

Thought: I have calculated the function's value at the interval boundaries and found the maximum of function $f(x)$ on interval $-1 \\leq x \\leq 3$ is 9.0.

Final Answer: $9$""",
    # example 2
    """Question: Olivia has $23. She bought five bagels for $3 each. How much money does she 
have left?
    
Thought: Olivia has $23 and bought five bagels for $3 each. To compute how much money 
left, I need to calculate the total cost of the bagels and then subtract it from the 
amount Olivia has to find out how much money she has left.

Action: python_interpreter

Action Input: ```python
total_cost = 5 * 3
money_left = 23 - total_cost
print(money_left)
```

Observation: 8

Thought: Olivia has $8 left after buying 5 bagels for $3 each.

Final Answer: $8$""",
]

math_examples = [
    # example 1
    """Question: Given the complex number $z=\\frac{{\\sqrt{{3}}+i}}{{(1-\\sqrt{{3}} i)^{{2}}}}$, where $\\bar{{z}}$ is the conjugate complex number of $z$, then calculate $z\\cdot\\bar{{z}}$.

Thought: The task is to find the produc of a complex and its conjugate. 
The problem has already been simplified to finding the square root of the magnitude of the complex number. 
So we need to calculate the magnitude of the complex number $z=\\frac{{\\sqrt{{3}}+i}}{{(1-\\sqrt{{3}} i)^{{2}}}}$.

Action: python_interpreter

Action Input: ```python
from sympy import *

z = (sqrt(3) + I) / ((1 - sqrt(3)*I)**2)
magnitude_z = abs(z)
print(magnitude_z)
```

Observation: 1/2

Thought: From the result, we can see that the magnitude of z is $\\frac{{1}}{{2}}$. 
The product of a complex number and its conjugate is equal to the square of the magnitude of the complex. 
So, to find $z\\cdot\\bar{{z}}$, we square the magnitude of z.

Action: python_interpreter

Action Input: ```python
z_conj_product = magnitude_z**2
print(z_conj_product)
```

Observation: 1/4

Thought: From the result, we can see that $z \cdot \\bar{{z}}$ is equal to $\\frac{{1}}{{4}}$.

Final Answer: $\\frac{{1}}{{4}}$""",
    # example 2
    """Question: Compute $\\arctan ( \\tan 65^\\circ - 2 \\tan 40^\\circ )$.

Thought: To solve this problem, the key is to simplify the expression $\\tan 65^\\circ - 2 \\tan 40^\\circ$. 
By applying the complementary angle identity $\\tan (90^\\circ - x) = \\frac{{1}}{{\\tan x}}$ to $\\tan 65^\\circ$ and $\\tan 40^\\circ$, we can compute their complementary angles.

Action: python_interpreter

Action Input: ```python
from sympy import symbols, solve

x, y = symbols('x y')
eq1 = Eq(90 - x, 65)
eq2 = Eq(90 - y, 40)
solution = solve([eq1, eq2], [x, y])
print(solution)
```

Observation: {{x: 25, y: 50}}

Thought: According to the result, we have $\\tan 65^\\circ = \\frac{{1}}{{\\tan 25^\\circ}}$ and $\\tan 40^\\circ = \\frac{{1}}{{\\tan 50^\\circ}}$. 
Substituting these in the expression, we get \[\\tan 65^\\circ - 2 \\tan 40^\\circ = \\frac{{1}}{{\\tan 25^\\circ}} - \\frac{{2}}{{\\tan 50^\\circ}}.\]. 
By the double-angle formula of tangent $\\tan(2x) = \\frac{{2 \\tan x}}{{1 - \\tan^2 x}}$, we apply it to $\\tan 50^\\circ$. 
Then, wen can simplify the expression by considering the $\\tan 25^\\circ$ as a variable.

Action: python_interpreter

Action Input: ```python
from sympy import simplify

# define variable
tan25 = symbols('tan25')

# double-angle formula of tangent
tan50 = 2*tan25 / (1 - tan25**2)

solution = simplify(1 / tan25 - 2 / tan50)
print(solution)
```

Observation: tan25

Thought: From the result, we know that $\\tan 65^\\circ - 2 \\tan 40^\\circ = \\tan 25^\\circ$. So $\\arctan (\\tan 25^\\circ) = 25^\\circ.$

Final Answer: $25^\\circ$""",
]
