import torch
import lightning as L
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2


class LitClassification(L.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        optimizer: str = "adamw",
        lr: float = 1e-3,
        scheduler: str = "cosine",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.backbone = backbone
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=200)
        self.val_acc = Accuracy(task="multiclass", num_classes=200)
        self.test_acc = Accuracy(task="multiclass", num_classes=200)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_best_loss = MinMetric()

    def forward(self, x) -> torch.Tensor:
        out_tensor = self.backbone(x)
        return out_tensor

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            model_or_params=self.backbone.parameters(),
            opt=self.hparams.optimizer,
            lr=self.hparams.lr,
        )

        scheduler = create_scheduler_v2(
            optimizer=optimizer,
            sched=self.hparams.scheduler,
        )
        return [optimizer], [scheduler]

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_best_loss.reset()

    def training_step(self, batch, batch_idx):
        loss, predicted_label, label = self._step(batch)
        self.train_acc(predicted_label, label)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        loss, predicted_label, label = self._step(batch)
        self.val_acc(predicted_label, label)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        self.val_best_loss(loss)
        self.log("val/best_loss", self.val_best_loss, on_step=False, on_epoch=True)

    def _step(self, batch):
        image, label = batch
        logits = self.forward(image)
        loss = self.criterion(logits, label)
        predicted_label = logits.argmax(dim=1)

        return loss, predicted_label, label
