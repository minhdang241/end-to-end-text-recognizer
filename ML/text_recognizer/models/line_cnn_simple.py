from typing import Any, Dict
import argparse
import math

import torch
import torch.nn as nn

from .cnn import CNN

CONV_DIM = 64
FC_DIM = 128
WINDOW_WIDTH = 28
WINDOW_STRIDE = 28


class LineCNNSimple(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.WW = self.args.get("window_width", WINDOW_WIDTH)
        self.WS = self.args.get("window_stride", WINDOW_STRIDE)
        self.limit_output_length = self.args.get("limit_output_length", False)

        self.num_classes = len(data_config["mapping"])
        self.output_length = data_config["output_dims"][0]
        self.cnn = CNN(data_config=data_config, args=args)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @params:
            x: input image of shape (B,1,H,W)
        @returns:
            torch.Tensor of shape (B,C,S), where S is the sequence length, C is the number of classes, and B is the batch size
        """
        B, _C, H, W = x.shape
        #assert H == IMAGE_SIZE

        S = math.floor((W - self.WW)/self.WS + 1)
        activations = torch.zeros((B, self.num_classes, S)).as_type(x)

        for s in range(S):
            start_w = self.WS * s
            end_w = start_w + self.WW
            activations[:,:,s] = self.cnn(x[:,:,:,start_w:end_w])
        
        if self.limit_output_length:
            activations = activations[:, :, :, self.output_length]
        
        return activations

    @staticmethod
    def add_to_argparse(parser):
        CNN.add_to_argparse(parser)
        parser.add_argument(
            "--window_width",
            type=int,
            default=WINDOW_WIDTH,
            help="Width of the window taht will slide over the input"
        )

        parser.add_argument(
            "--window_stride",
            type=int, 
            default=WINDOW_STRIDE,
            help="Stride of the window that will slide over the input image"
        )

        parser.add_argument("--limit_output_length", action="store_true", default=False)
        return parser