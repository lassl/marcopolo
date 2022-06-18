import logging

import datasets
import torch
import torch.distributed as dist
import transformers


def set_logging_default(logger, accelerator):

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.ERROR)
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.disable_progress_bar()
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_divisibility(numerator, denominator):
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    check_divisibility(numerator, denominator)
    return numerator // denominator


def _split(tensor, dim=-1):
    if dist.get_world_size() == 1:
        return tensor

    split_size = divide(tensor.shape[dim], dist.get_world_size())
    tensor_list = torch.split(tensor, split_size, dim=dim)

    output = tensor_list[dist.get_rank()].contiguous()
    return output


def _gather(tensor, dim=-1):
    if dist.get_world_size() == 1:
        return tensor
    tensor_list = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor, async_op=False)

    output = torch.cat(tensor_list, dim=dim)
    return output


def gather(input, dim=-1):
    if dist.is_initialized():
        if torch.is_grad_enabled() and input.requires_grad:
            input = Gather.apply(input, dim)
        else:
            input = _gather(input, dim=dim)
    return input


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim: int = -1):
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        (dim,) = ctx.saved_tensors
        return _split(grad_output, dim=int(dim)), None
