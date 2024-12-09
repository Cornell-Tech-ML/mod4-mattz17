from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for a given variable.

        This method updates the internal state to include the derivative
        corresponding to the input `x`. It is typically used during
        backpropagation to aggregate gradients for optimization.

        Args:
        ----
            x (Any): The variable or value for which the derivative is being
                    accumulated.

        Returns:
        -------
            None: This method does not return a value; it modifies the internal
                state of the object.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique id of the variable"""
        ...

    def is_leaf(self) -> bool:
        """Return True if this variable is a leaf, False otherwise"""
        ...

    def is_constant(self) -> bool:
        """Return True if this variable is a constant, False otherwise"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of this variable"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the derivatives of the output with respect to the input variables using the chain rule.

        Args:
        ----
        d_output (Any): The derivative of the output with respect to the loss
                        or next layer, used to compute the gradients.


        Returns:
        -------
        Iterable[Tuple[Variable, Any]]: A list of tuples, each pairing a
                                          non-constant input variable with its
                                          corresponding derivative.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieve the saved tensor values.

        This method returns a tuple of tensor values that have been saved during
        the forward pass or computation. These values are typically used for
        later operations, such as backpropagation.

        Returns
        -------
            Tuple[Any, ...]: A tuple containing the saved tensor values.

        """
        return self.saved_values
