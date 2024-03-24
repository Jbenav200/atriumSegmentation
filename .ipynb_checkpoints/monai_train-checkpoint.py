from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt

from MONAICardiacDataset import MONAICardiacDataset
from MONAIModel import UNet
import monai

seq = iaa.Sequential([
    iaa.Affine(scale=(0.85, 1.15),
              rotate=(-45, 45)),
    iaa.ElasticTransformation()
])

# Create the dataset objects
train_path = Path("Preprocessed/train/")
val_path = Path("Preprocessed/val")

train_dataset = MONAICardiacDataset(train_path, seq)
val_dataset = MONAICardiacDataset(val_path, None)

print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")

class AtriumSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = UNet()
        
        self.optimizer = monai.optimizers.Novograd(self.model.parameters(), lr=1e-4)
        self.loss_fn = monai.losses.DiceLoss()
        
    def forward(self, data):
        return torch.sigmoid(self.model(data))
    
    def training_step(self, batch, batch_idx):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)
        loss = self.loss_fn(pred, mask)
#         hausdorff = monai.metrics.compute_hausdorff_distance(pred, mask)
#         prec = monai.metrics.compute_confusion_matrix_metric(
#             metric_name="precision",
#             confusion_matrix=monai.metrics.get_confusion_matrix(pred, mask)
#         )
#         rec = monai.metrics.compute_confusion_matrix_metric(
#             metric_name="recall",
#             confusion_matrix=monai.metrics.get_confusion_matrix(pred, mask)
#         )
        
        self.log("Train Dice", loss)
#         self.log("Train Hausdorff Distance", hausdorff.mean())
#         self.log("Train Precision", prec.mean())
#         self.log("Train Recall", rec.mean())
        
        if batch_idx % 50 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Train")
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)
        loss = self.loss_fn(pred, mask)
#         hausdorff = monai.metrics.compute_hausdorff_distance(pred, mask)
#         prec = monai.metrics.compute_confusion_matrix_metric(
#             metric_name="precision",
#             confusion_matrix=monai.metrics.get_confusion_matrix(pred, mask)
#         )
#         rec = monai.metrics.compute_confusion_matrix_metric(
#             metric_name="recall",
#             confusion_matrix=monai.metrics.get_confusion_matrix(pred, mask)
#         )
        
        self.log("Val Dice", loss)
#         self.log("Val Hausdorff Distance", hausdorff.mean())
#         self.log("Val Precision", prec.mean())
#         self.log("Val Recall", rec.mean())
        
        if batch_idx % 2 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Val")
            
        return loss
    
    def log_images(self, mri, pred, mask, name):
        
        pred = pred > 0.5
        
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(mri[0][0], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][0]==0, mask[0][0])
        axis[0].imshow(mask_, alpha=0.6)
        
        axis[1].imshow(mri[0][0], cmap="bone")
        mask_ = np.ma.masked_where(pred[0][0]==0, pred[0][0])
        axis[1].imshow(mask_, alpha=0.6)
        
        self.logger.experiment.add_figure(name, fig, self.global_step)
        
    def configure_optimizers(self):
        return [self.optimizer]
    


if __name__ == "__main__":
    batch_size = 8
    num_workers = 10

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    torch.manual_seed(0)
    model = AtriumSegmentation()
    # Create the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='Val Dice',
        save_top_k=10,
        mode='min')
    trainer = pl.Trainer(gpus=1, logger=TensorBoardLogger(save_dir='logs'), log_every_n_steps=1,
                    callbacks=checkpoint_callback, max_epochs=75, accelerator='mps')
    
    trainer.fit(model, train_loader, val_loader)