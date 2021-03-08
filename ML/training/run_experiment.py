import argparse
import importlib

import numpy as np
import torch 
import pytorch_lightning as pl
import pprint
from text_recognizer import lit_models

# To ensure reproducibility
pl.seed_everything(24)

def _import_class(module_and_class_name: str) -> type:
    """Import class from a module
    @params: module_and_class_name: e.g text_recognizer.models.cnn
    @returns: the class object
    """
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    "Set up Argument Parser"
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer's arguments
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--data_class", type=str, default="MNIST")
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model class to add extra
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"text_recognizer.data.{temp_args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{temp_args.model_class}")

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment
    """

    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"text_recognizer.data.{args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    if args.loss not in ('ctc', 'transformer'):
        lit_model_class = lit_models.BaseLitModel
    elif args.loss == "ctc":
        lit_model_class = lit_models.CTCLitModel
    elif args.loss == "transformer":
        lit_model_class = lit_models.TransformerLitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)
    
    logger = pl.loggers.TensorBoardLogger("training/logs")

    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))
    
    callbacks = [pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)]
    args.weights_summary = "full"
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs")
    trainer.tune(lit_model, datamodule=data)
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)

if __name__ == "__main__":
    main()