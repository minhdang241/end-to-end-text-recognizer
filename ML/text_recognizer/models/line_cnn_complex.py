from typing import Any, Dict
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_DIMS= [64, 128, 128, 256, 256, 512, 512, 512]
FC_DIM = 512


class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out


class LineCNNComplex(nn.Module):
    """
    Model that uses a simple CNN to process an image of a line of characters with a window, outputting a sequence of logits.
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}
        self.num_classes = len(data_config["mapping"])
        self.output_length = data_config["output_dims"][0]

        _C, H, _W = data_config["input_dims"]
        conv_dims = self.args.get("conv_dims", CONV_DIMS)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        self.fc_dim = fc_dim

        # Input is (1, H, W)
        convs = []
        for i, conv_dim in enumerate(conv_dims):
            if i == 0:
                convs.append(ConvBlock(1, conv_dim))
                convs.append(nn.MaxPool2d(2))
            elif i in [2, 4, 6]:
                convs.append(ConvBlock(conv_dims[i-1], conv_dim, kernel_size=3))                
                convs.append(nn.MaxPool2d(2))           
            else:
                convs.append(ConvBlock(conv_dims[i-1], conv_dim))

        # convs.append(ConvBlock(512, 512, kernel_size=(2,3)))
        self.convs = nn.Sequential(*convs)
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(fc_dim, self.num_classes)

        self._init_weights()

    def _init_weights(self):
        """
        A better weight initialization scheme than PyTorch default.

        See https://github.com/pytorch/pytorch/issues/18182
        """
        for m in self.modules():
            if type(m) in {
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            }:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, 1, H, W) input image

        Returns
        -------
        torch.Tensor
            (B, C, S) logits, where S is the length of the sequence and C is the number of classes
            S can be computed from W and self.window_width
            C is self.num_classes
        """
        _B, _C, _H, W = x.shape
        x = self.convs(x)  # (B, FC_DIM, 1, Sx)
        x = x.reshape(_B, self.fc_dim, -1).permute(0, 2, 1)  # (B, S, FC_DIM)
        x = F.relu(self.fc1(x))  # -> (B, S, FC_DIM)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, S, C)
        x = x.permute(0, 2, 1)  # -> (B, C, S)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dims", type=int, default=CONV_DIMS)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--limit_output_length", action="store_true", default=False)
        return parser
