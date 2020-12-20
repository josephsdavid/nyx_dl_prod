import torch
from typing import Dict, Set
from collections import Counter
from itertools import chain

Model = torch.nn.Module
ParamDict = Dict[str, torch.tensor]
StateDiff = Set[str]


def state_dict_diff(m: Model, p: ParamDict) -> StateDiff:
    return set(m.state_dict().keys()) - set(p.keys())

def count_parameters(m: Model) -> Counter:
    return Counter(chain(*((p.requires_grad for p in child.parameters()) for child in m.children())))
