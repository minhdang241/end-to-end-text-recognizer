import argparse
import itertools
import torch

from .base import BaseLitModel
from .metrics import CharacterErrorRate
from .util import first_element, ctcBeamSearch
import wandb

def compute_input_lengths(padded_sequences: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    padded_sequences
        (N, S) tensor where elements that equal 0 correspond to padding

    Returns
    -------
    torch.Tensor
        (N,) tensor where each element corresponds to the non-padded length of each sequence

    Examples
    --------
    >>> X = torch.tensor([[1, 2, 0, 0, 0], [1, 2, 3, 0, 0], [1, 2, 3, 0, 5]])
    >>> compute_input_lengths(X)
    tensor([2, 3, 5])
    """
    lengths = torch.arange(padded_sequences.shape[1]).type_as(padded_sequences)
    return ((padded_sequences > 0) * lengths).argmax(1) + 1


class CTCLitModel(BaseLitModel):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args)

        self.inverse_mapping = {val: ind for ind, val in enumerate(self.model.data_config["mapping"])}
        # start_index = inverse_mapping["<S>"]
        self.blank_index = inverse_mapping["<B>"]
        # end_index = inverse_mapping["<E>"]
        self.padding_index = inverse_mapping["<P>"]

        # Save hyperparameters
        self.save_hyperparameters()

        self.loss_fn = torch.nn.CTCLoss(zero_infinity=True)

        # ignore_tokens = [start_index, end_index, self.padding_index]
        ignore_tokens = [self.padding_index, self.blank_index]
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=1e-3)
        return parser

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        B, _C, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # -> (S, B, C)

        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
        target_lengths = first_element(y, self.padding_index).type_as(y)
        loss = self.loss_fn(logprobs_for_loss, y, input_lengths, target_lengths)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        B, _C, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # -> (S, B, C)
        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S  # All are max sequence length
        target_lengths = first_element(y, self.padding_index).type_as(y)  # Length is up to first padding token
        loss = self.loss_fn(logprobs_for_loss, y, input_lengths, target_lengths)
        self.log("val_loss", loss, prog_bar=True)

        decoded = self.greedy_decode(logprobs, max_length=y.shape[1])
        self.val_acc(decoded, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_cer(decoded, y)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        decoded = self.greedy_decode(logprobs, max_length=y.shape[1])

        try:
            self.logger.experiment.log({
                "test_pred_examples": [wandb.Image(x[0], 
                caption=f"Pred:{convert_y_label_to_string(decoded[0])}, Label:{convert_y_label_to_string(y[0])}")]
            })
        except AttributeError:
            pass

        self.test_acc(decoded, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_cer(decoded, y)
        self.log("test_cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
    
    def convert_y_label_to_string(self, y):
        result = ''.join([self.mapping[i] for i in y if i != self.inverse_mapping["<P>"]])
        return result

    def greedy_decode(self, logprobs: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Greedily decode sequences, collapsing repeated tokens, and removing the CTC blank token.

        See the "Inference" sections of https://distill.pub/2017/ctc/

        Using groupby inspired by https://github.com/nanoporetech/fast-ctc-decode/blob/master/tests/benchmark.py#L8

        Parameters
        ----------
        logprobs
            (B, C, S) log probabilities
        max_length
            max length of a sequence

        Returns
        -------
        torch.Tensor
            (B, S) class indices
        """
        B = logprobs.shape[0]
        argmax = logprobs.argmax(1)
        decoded = torch.ones((B, max_length)).type_as(logprobs).int() * self.padding_index
        for i in range(B):
            seq = [b for b, _g in itertools.groupby(argmax[i].tolist()) if b != self.blank_index][:max_length]
            for ii, char in enumerate(seq):
                decoded[i, ii] = char
        return decoded

    # def beam_search(self, logprobs: torch.Tensor, max_length: int) -> torch.Tensor:
    #     """
    #     @params: logprobs shape of (B, C, S)
    #     """
    #     B = logprobs.shape[0]
    #     decoded = torch.ones((B, max_length)).type_as(logprobs).int() * self.padding_index
    #     for i in range(B):
    #         seq = ctcBeamSearch(logprobs[i], self.padding_index, max_length, None) 
    #         for ii, char in enumerate(seq):
    #             decoded[i, ii] = char
    #     return decoded
