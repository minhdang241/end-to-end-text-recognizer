import pytorch_lightning as pl
import wandb
import torch
class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=32, mapping=None):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]
        self.mapping = mapping 

    def convert_y_label_to_string(self, y):
        return ''.join([self.mapping[i] for i in y if i != 3])

          
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                            for x, pred, y in zip(val_imgs, self.convert_y_label_to_string(preds), self.convert_y_label_to_string(self.val_labels))],
            "global_step": trainer.global_step
            })