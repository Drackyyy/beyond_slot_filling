import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

AVAIL_GPUS = min(1, torch.cuda.device_count())

class TaggingModel(LightningModule):

    def __init__(
        self,
        model_name_or_path: str = 'vblagoje/bert-english-uncased-finetuned-pos',
        num_labels: int = 3,
        task_name: str = 'salient_words_tagging',
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features=in_features,out_features=self.hparams.num_labels,bias=True)
        model.num_labels = self.hparams.num_labels
        self.model = model
        self.batch_size = train_batch_size


    def forward(self, **inputs):  ## later we need to define a hooker here to get embeddings.
        pass

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log('training_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(**batch)
        val_loss, logits = outputs[:2]
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=2)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        self.log('val_loss', val_loss, on_step=True, prog_bar=True, logger=True)
        return {'loss': val_loss, "preds": preds, "labels": labels}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(**batch)
        test_loss, logits = outputs[:2]
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=2)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        accuracy = sum(sum(labels==preds)).item()/len(labels)
        self.log('test_loss', test_loss, on_step=True, prog_bar=True, logger=True)
        self.log('test_accuracy', test_loss, on_step=True, prog_bar=True, logger=True)



    def setup(self, stage=None) -> None:
        if stage != 'fit':
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()
        # Calculate total steps
        try:
            tb_size = self.hparams.train_batch_size * max(1, len(self.trainer.gpus))
        except:
            tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus) 
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', loss, prog_bar=True, logger=False)
        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]