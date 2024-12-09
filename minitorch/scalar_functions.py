from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to a list of input values.

        Args:
        ----
            *vals (ScalarLike): A variable number of input values, which can
                                be either `Scalar` objects or raw numerical values.

        Returns:
        -------
            Scalar: A new `Scalar` object containing the result of the
                    function applied to the input values, with backtracking
                    history for future gradient computations.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward computation for the addition operation.

        This method computes the sum of two input values and stores any necessary
        context for backpropagation.

        Args:
        ----
            ctx (Context): The context object to store information for the
                        backward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of adding the two input values.

        """
        return float(a + b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradients for the inputs during the backward pass.

        This method returns the gradients of the inputs with respect to the
        output, based on the chain rule of calculus. For an addition operation,
        the gradient with respect to both inputs is equal to the incoming
        gradient.

        Args:
        ----
            ctx (Context): The context object used in the forward pass (not
                        used in this implementation).
            d_output (float): The gradient of the output with respect to the
                            loss or next layer.

        Returns:
        -------
            Tuple[float, ...]: A tuple containing the gradients for each input
                                (d_output, d_output) for the two inputs of the
                                addition operation.

        """
        return float(d_output), float(d_output)


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for Log

        Args:
        ----
            ctx: Context object for storing variables.
            a: The operand.

        Returns:
        -------
            The forward pass of Log.

        """
        ctx.save_for_backward(a)
        return float(operators.log(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for Log.

        Args:
        ----
            ctx: Context object for storing variables.
            d_output: Derivative of the output.

        Returns:
        -------
            The backward pass of Log.

        """
        (a,) = ctx.saved_values
        return float(operators.log_back(a, d_output))


# To implement.


# TODO: Implement for Task 1.2.
class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication.

        Args:
        ----
            ctx: Context object for storing variables.
            a: First operand.
            b: Second operand.

        Returns:
        -------
            The forward pass of multiplication.

        """
        ctx.save_for_backward(a, b)
        return float(a * b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx: Context object for storing variables.
            d_output: Derivative of the output.

        Returns:
        -------
            The backward pass of multiplication.

        """
        (a, b) = ctx.saved_values
        return float(b * d_output), float(a * d_output)


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse.

        Args:
        ----
            ctx: Context object for storing variables.
            a: Operand.

        Returns:
        -------
            The forward pass of inverse.

        """
        ctx.save_for_backward(a)
        return float(operators.inv(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse.

        Args:
        ----
            ctx: Context object for storing variables.
            d_output: Derivative of the output.

        Returns:
        -------
            The backward pass of inverse.

        """
        (a,) = ctx.saved_values
        return float(operators.inv_back(a, d_output))


class Neg(ScalarFunction):
    """Neg function $f(x) = -f(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse.

        Args:
        ----
            ctx: Context object for storing variables.
            a: Operand.

        Returns:
        -------
            The forward pass of inverse.

        """
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse.

        Args:
        ----
            ctx: Context object for storing variables.
            d_output: Derivative of the output.

        Returns:
        -------
            The backward pass of inverse.

        """
        return float(-d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid.

        Args:
        ----
            ctx: Context object for storing variables.
            a: Operand.

        Returns:
        -------
            The forward pass of inverse.

        """
        x = float(operators.sigmoid(a))
        ctx.save_for_backward(x)
        return float(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid.

        Args:
        ----
            ctx: Context object for storing variables.
            d_output: Derivative of the output.

        Returns:
        -------
            The backward pass of sigmoid.

        """
        (a,) = ctx.saved_values
        return float(d_output * a * (1 - a))


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for Relu.

        Args:
        ----
            ctx: Context object for storing variables.
            a: Operand.

        Returns:
        -------
            The forward pass of inverse.

        """
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for relu.

        Args:
        ----
            ctx: Context object for storing variables.
            d_output: Derivative of the output.

        Returns:
        -------
            The backward pass of relu.

        """
        (a,) = ctx.saved_values
        return float(operators.relu_back(a, d_output))


class Exp(ScalarFunction):
    """Exp function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exp.

        Args:
        ----
            ctx: Context object for storing variables.
            a: Operand.

        Returns:
        -------
            The forward pass of inverse.

        """
        a = float(operators.exp(a))
        ctx.save_for_backward(a)
        return a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exp.

        Args:
        ----
            ctx: Context object for storing variables.
            d_output: Derivative of the output.

        Returns:
        -------
            The backward pass of exp.

        """
        (a,) = ctx.saved_values
        return float(d_output * a)


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for lt.

        Args:
        ----
            ctx: Context object for storing variables.
            a: Operand.
            b: Operand.

        Returns:
        -------
            The forward pass of lt.

        """
        return float(operators.lt(a, b))  # Return float instead of bool

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for lt.

        Args:
        ----
            ctx: Context object for storing variables.
            d_output: Derivative of the output.

        Returns:
        -------
            The backward pass of lt.

        """
        return 0.0, 0.0  # The gradient for comparison ops is typically 0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for eq.

        Args:
        ----
            ctx: Context object for storing variables.
            a: Operand.
            b: Operand.

        Returns:
        -------
            The forward pass of eq.

        """
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for eq.

        Args:
        ----
            ctx: Context object for storing variables.
            d_output: Derivative of the output.

        Returns:
        -------
            The backward pass of eq.

        """
        return 0.0, 0.0
