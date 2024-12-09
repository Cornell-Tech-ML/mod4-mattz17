"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(a: float, b: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        a: a number
        b: a number

    Returns:
    -------
        The product of a and b.

    """
    return a * b


# - id
def id(a: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        a: a number

    Returns:
    -------
        The identity of a.

    """
    return a


# - add
def add(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
    ----
        a: a number
        b: a number

    Returns:
    -------
        The sum of a and b.

    """
    return a + b


# - neg
def neg(a: float) -> float:
    """Negates a number.

    Args:
    ----
        a: a number

    Returns:
    -------
        The negation of a.

    """
    return -a


# - lt
def lt(a: float, b: float) -> bool:
    """Checks if a is less than b.

    Args:
    ----
        a: a number
        b: a number

    Returns:
    -------
        True if a is less than b, False otherwise.

    """
    return a < b


# - eq
def eq(a: float, b: float) -> bool:
    """Checks if a is equal to b.

    Args:
    ----
        a: a number
        b: a number

    Returns:
    -------
        True if a is equal to b, False otherwise.

    """
    return a == b


# - max
def max(a: float, b: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        a: a number
        b: a number

    Returns:
    -------
        The larger of a and b.

    """
    return a if a > b else b


# - is_close
def is_close(a: float, b: float) -> bool:
    """Checks if a is close in value to b.

    Args:
    ----
        a: a number
        b: a number

    Returns:
    -------
        True if a is close in value to b, False otherwise.

    """
    return abs(a - b) < 1e-2


# - sigmoid
def sigmoid(a: float) -> float:
    """Calculates the sigmoid of a.

    Args:
    ----
        a: a number

    Returns:
    -------
        The sigmoid of a.

    """
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1.0 + math.exp(a))


# - relu
def relu(a: float) -> float:
    """Calculates the ReLU of a.

    Args:
    ----
        a: a number

    Returns:
    -------
        The ReLU of a.

    """
    return a if a > 0 else 0.0


EPS = 1e-6


# - log
def log(a: float) -> float:
    """Calculates the natural logarithm of a.

    Args:
    ----
        a: a number

    Returns:
    -------
        The natural logarithm of a.

    """
    return math.log(a + EPS)


# - exp
def exp(a: float) -> float:
    """Calculates the exponential of a.

    Args:
    ----
        a: a number

    Returns:
    -------
        The exponential of a.

    """
    return math.exp(a)


# - log_back
def log_back(a: float, b: float) -> float:
    """Calculates the derivative of the logarithm of a with respect to b.

    Args:
    ----
        a: a number
        b: a number

    Returns:
    -------
        The derivative of the logarithm of a with respect to b.

    """
    return b / (a + EPS)


# - inv
def inv(a: float) -> float:
    """Calculates the inverse of a.

    Args:
    ----
        a: a number

    Returns:
    -------
        The inverse of a.

    """
    return 1.0 / a


# - inv_back
def inv_back(a: float, b: float) -> float:
    """Calculates the derivative of the inverse of a with respect to b.

    Args:
    ----
        a: a number
        b: a number

    Returns:
    -------
        The derivative of the inverse of a with respect to b.

    """
    return -1.0 / (a * a) * b


# - relu_back
def relu_back(a: float, b: float) -> float:
    """Calculates the derivative of the ReLU of a with respect to b.

    Args:
    ----
        a: a number
        b: a number

    Returns:
    -------
        The derivative of the ReLU of a with respect to b.

    """
    if a > 0:
        return b
    else:
        return 0.0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a function to each element of a list.

    Args:
    ----
        f: a function

    Returns:
    -------
        A list of the results of applying f to each element of ls.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(f(x))
        return ret

    return _map


# - zipWith
def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Applies a function to corresponding elements of two lists.

    Args:
    ----
        f: a function

    Returns:
    -------
        A list of the results of applying f to each pair of elements from ls1 and ls2.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(f(x, y))
        return ret

    return _zipWith


# - reduce
def reduce(
    f: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduces a list to a single value by applying a function cumulatively.

    Args:
    ----
        f: a function
        start: a number

    Returns:
    -------
        The result of applying f cumulatively to the elements of ls.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = f(val, l)
        return val

    return _reduce


#
# Use these to implement
# - negList : negate a list
def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negates a list.

    Args:
    ----
        ls: a list

    Returns:
    -------
        A list of the negated elements of ls.

    """
    return map(neg)(ls)


# - addLists : add two lists together
def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Adds two lists together.

    Args:
    ----
        ls1: a list
        ls2: a list

    Returns:
    -------
        A list of the sums of the corresponding elements of ls1 and ls2.

    """
    return zipWith(add)(ls1, ls2)


# - sum: sum lists
def sum(ls: Iterable[float]) -> float:
    """Sums a list.

    Args:
    ----
        ls: a list

    Returns:
    -------
        The sum of the elements of ls.

    """
    return reduce(add, 0.0)(ls)


# - prod: take the product of lists
def prod(ls: Iterable[float]) -> float:
    """Takes the product of a list.

    Args:
    ----
        ls: a list

    Returns:
    -------
        The product of the elements of ls.

    """
    return reduce(mul, 1.0)(ls)
