"""Base Dataset class."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch


SequenceOrTensor = Union[Sequence, torch.Tensor] # either Sequence or torch.Tensor


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class processes data and targets through optional transforms

    @params: 
        data: torch tensors, numpy arrays, or PIL images
        targets: torch tensors or numpy arrays
        transform: function that takes a datum and return the same
        target_transform: function that takes a target and return the same
    """

    def __init__(self, data: SequenceOrTensor, targets: SequenceOrTensor,
                 transform: Callable = None, target_transform: Callable = None) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and target must be of equal length")
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    
    def __len__(self):
        """Return length of the dataset"""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.
        @params: index
        @return: (datum, target)
        """
        datum, target = self.data[index], self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target
    

def convert_strings_to_labels(strings: Sequence[str], mapping: Dict[str, int], length: int) -> torch.Tensor:
    """
    Convert sequence of N strings to a (N, length) narray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token
    """

    labels = torch.ones((len(strings), length), dtype=torch.long)*mapping["<P>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]
        for j, token in enumerate(tokens):
            labels[i][j] = mapping(token)
    return labels