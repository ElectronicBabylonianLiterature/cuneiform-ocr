from typing import List, Any

import torch
import torch.nn.functional as F

BaseDataElement = Any



def batch_label_to_onehot(batch_label, split_indices, num_classes):
    """Convert a concated label tensor to onehot format.

    Args:
        batch_label (torch.Tensor): A concated label tensor from multiple
            samples.
        split_indices (List[int]): The split indices of every sample.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: The onehot format label tensor.

    Examples:
    """
    sparse_onehot_list = F.one_hot(batch_label, num_classes)
    onehot_list = [
        sparse_onehot.sum(0)
        for sparse_onehot in tensor_split(sparse_onehot_list, split_indices)
    ]
    return torch.stack(onehot_list)


def cat_batch_labels(elements: List[torch.Tensor]):
    """Concat a batch of label tensor to one tensor.

    Args:
        elements (List[tensor]): A batch of labels.

    Returns:
        Tuple[torch.Tensor, List[int]]: The first item is the concated label
        tensor, and the second item is the split indices of every sample.
    """
    labels = []
    splits = [0]
    for element in elements:
        labels.append(element)
        splits.append(splits[-1] + element.size(0))
    batch_label = torch.cat(labels)
    return batch_label, splits[1:-1]


if hasattr(torch, 'tensor_split'):
    tensor_split = torch.tensor_split
else:
    # A simple implementation of `tensor_split`.
    def tensor_split(input: torch.Tensor, indices: list):
        outs = []
        for start, end in zip([0] + indices, indices + [input.size(0)]):
            outs.append(input[start:end])
        return outs