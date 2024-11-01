import torch
import evaluate
import lightning as L
from datetime import datetime
from typing import Optional
from torch.optim import AdamW, Adam, SGD
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)


class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        weight_decay: float = 1e-5,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )
        self.metric = evaluate.load(
            "glue",
            self.hparams.task_name,
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        )

        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append(
            {"loss": val_loss, "preds": preds, "labels": labels}
        )
        return val_loss

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v
                    for k, v in self.metric.compute(
                        predictions=preds, references=labels
                    ).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            self.validation_step_outputs.clear()
            return loss

        preds = (
            torch.cat([x["preds"] for x in self.validation_step_outputs])
            .detach()
            .cpu()
            .numpy()
        )
        labels = (
            torch.cat([x["labels"] for x in self.validation_step_outputs])
            .detach()
            .cpu()
            .numpy()
        )
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(
            self.metric.compute(predictions=preds, references=labels), prog_bar=True
        )
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            }
        ]

        optimizer = None
        # Initialize optimizer based on the provided hyperparameter
        if self.hparams.optimizer == "AdamW":
            optimizer_grouped_parameters[0]["eps"] = self.hparams.epsilon
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer == "Adam":
            optimizer_grouped_parameters[0]["eps"] = self.hparams.epsilon
            optimizer = Adam(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer == "SGD":
            optimizer_grouped_parameters[0]["momentum"] = self.hparams.momentum
            optimizer = SGD(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {self.hparams.optimizer}")

        # Initialize learning rate scheduler
        scheduler = None
        if self.hparams.scheduler == "linearSchedule":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.scheduler == "cosineSchedule":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.scheduler == "constantSchedule":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.hparams.warmup_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.hparams.scheduler}")

        scheduler_dict = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler_dict]
