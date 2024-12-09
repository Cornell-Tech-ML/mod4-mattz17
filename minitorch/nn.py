from typing import Tuple


from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    tiled = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    tiled = tiled.permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform average pooling on a 2D tensor.

    Args:
        input (Tensor): Input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): Pooling kernel size as (height, width).

    Returns:
        Tensor: Pooled tensor of shape (batch, channel, new_height, new_width).

    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    tiled, new_height, new_width = tile(input, kernel)

    # Reshape to ensure compatibility with indexing
    reshaped_tiled = tiled.contiguous()

    # Compute the average along the kernel dimensions
    pooled = reshaped_tiled.mean(dim=-1)
    pooled = pooled.view(batch, channel, pooled.shape[2], pooled.shape[3])

    return pooled


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a one-hot tensor along a specified dimension.

    Args:
        input (Tensor): Input tensor.
        dim (int): Dimension along which to compute the argmax.

    Returns:
        Tensor: A tensor with the same shape as the input, containing 1 at the position of the maximum value along the specified dimension and 0 elsewhere.

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        # TODO: Implement for Task 4.4.
        max_red = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, max_red)
        return max_red

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        # TODO: Implement for Task 4.4.
        (input, max_red) = ctx.saved_values
        return (grad_output * (max_red == input)), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the maximum value along a specified dimension.

    Args:
        input (Tensor): Input tensor.
        dim (int): Dimension along which to compute the maximum.

    Returns:
        Tensor: Tensor containing the maximum values along the specified dimension.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of a tensor along a specified dimension.

    Args:
        input (Tensor): Input tensor.
        dim (int): Dimension along which to compute the softmax.

    Returns:
        Tensor: Softmax tensor with probabilities along the specified dimension.

    """
    # TODO: Implement for Task 4.4.
    numerator = input.exp()
    denominator = numerator.sum(dim)
    return numerator / denominator


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log softmax of a tensor along a specified dimension.

    Args:
        input (Tensor): Input tensor.
        dim (int): Dimension along which to compute the log softmax.

    Returns:
        Tensor: Log softmax tensor along the specified dimension.

    """
    # TODO: Implement for Task 4.4.
    max_values = max(input, dim)

    # Shift the input by subtracting the max value (numerical stability)
    shifted_input = input - max_values

    # Compute logsumexp along the specified dimension
    logsumexp = shifted_input.exp().sum(dim).log()

    # Compute the log softmax
    log_softmax = shifted_input - logsumexp

    return log_softmax


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform max pooling on a 2D tensor.

    Args:
        input (Tensor): Input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): Pooling kernel size as (height, width).

    Returns:
        Tensor: Pooled tensor of shape (batch, channel, new_height, new_width).

    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    tiled, new_height, new_width = tile(input, kernel)

    # Reshape to ensure compatibility with indexing
    reshaped_tiled = tiled.contiguous()

    # Compute the average along the kernel dimensions
    pooled = max(reshaped_tiled, dim=-1)
    pooled = pooled.view(batch, channel, pooled.shape[2], pooled.shape[3])

    return pooled


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to a tensor by randomly setting elements to zero.

    Args:
        input (Tensor): Input tensor.
        rate (float): Dropout rate (fraction of elements to drop).
        ignore (bool): If True, skip applying dropout.

    Returns:
        Tensor: Tensor after applying dropout, or the input tensor if ignore is True.

    """
    # TODO: Implement for Task 4.4.
    if ignore:
        return input
    else:
        randoms = rand(input.shape)
        dropped = randoms > rate  # type: ignore
        return input * dropped
