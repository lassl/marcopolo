import torch
from torch import Tensor
from typing import Optional, Any, List, Union


@torch.no_grad()
def gather_step_tensor(result: Tensor, group: Optional[Any] = None) -> List[Any]:
    """Function to gather all losses from several ddp processes onto a list that is broadcasted to all processes.
    ref_from: https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/distributed.html

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)

    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]

    # sync and broadcast all -> pick tensors item
    torch.distributed.barrier(group=group)
    torch.distributed.all_gather(gathered_result, result, group)

    return gathered_result


@torch.no_grad()
def get_idx_rank(scores: Tensor, labels: Tensor):
    return (labels.unsqueeze(1) == scores.argsort(dim=-1, descending=True)).nonzero(as_tuple=True)[1]


class MrrMetric:
    def __init__(self, topk: int = 10):
        self.topk = topk
        self.mrr_list: List[float] = []
        self.topk_elem: int = 0

    def clear(self):
        self.mrr_list: List[float] = []
        self.topk_elem: int = 0

    def update(self, input: Union[int, List[int]]):
        if isinstance(input, int):
            self.topk_elem += 1 if input < self.topk else 0
            mrr = (self.topk - input) / self.topk if input < self.topk else 0
            self.mrr_list.append(mrr)
        else:
            for item in input:
                self.topk_elem += 1 if item < self.topk else 0
                mrr = (self.topk - item) / self.topk if item < self.topk else 0
                self.mrr_list.append(mrr)

    def compute(self) -> float:
        return sum(self.mrr_list) / len(self.mrr_list) if len(self.mrr_list) > 0 else 0
