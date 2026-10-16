from typing import Any, Dict

from hydra.utils import call, instantiate
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from .task import Task, TaskStage
from ..models import GNN
from ..utils import Seeder


class BootstrapTask(Task, pl.LightningModule):
    def __init__(self, args: DictConfig):
        Task.__init__(self, args)
        self.task_stage = TaskStage.BOOTSTRAPPING
        pl.LightningModule.__init__(self)
        self.args.gnn.embeddings.model_name = args.model.hf_handle

        # Set seed for reproducability
        self.seeder = Seeder(args.seed)

        # INITIALIZE DATASET
        # Initialize tokenizer
        if "tokenizer" in self.args:
            self.tokenizer = call(
                self.args.tokenizer.type,
                self.args.tokenizer.path,
                **self.args.tokenizer.hyperparams,
            )
        else:
            self.tokenizer = call(
                {"_target_": "transformers.AutoTokenizer.from_pretrained"},
                self.args.model.hf_handle,
            )

    def configure_model(self):
        if self.has_initialized:
            return
        # This parameter is adaptive and calculated from target model tokenizer set
        self.args.model.vocab_size = len(self.tokenizer)

        # Propagate model related features, name set before to log correctly
        self.args.gnn.in_features = self.args.model.hidden_size
        self.args.gnn.out_features = self.args.model.hidden_size
        self.args.gnn.vocab_size = self.tokenizer.vocab_size

        # Initialize gnn model
        self.gnn: GNN = instantiate(self.args.gnn.type, self.args.gnn)

        # Add AST node embeddings to UniXcoder embedding space
        self.gnn.extend_embeddings(self.tokenizer, self.ast_types)

    def prepare_data(self) -> None:
        if self.has_initialized or getattr(self, "data_prepared", False):
            return
        # Propagate task definition to the Dataset
        self.args.dataset.language_task = self.args.language_task
        self.args.dataset.debug = self.args.debug
        self.args.dataset.add_graphs = True
        self.args.dataset.model_name = self.args.model.name

        # Initialize dataset
        self.dataset = instantiate(
            self.args.dataset.type, self.args.dataset, self.tokenizer, Seeder(self.args.seed)
        )

        # Extend the tokenizer with AST types such as 'method_declaration'
        if "unixcoder" in self.args.model.hf_handle:
            self.ast_types = GNN.extend_tokenizer(self.tokenizer)
        else:
            if self.dataset.syntax_nodes is not None:
                self.ast_types = GNN.extend_tokenizer(self.tokenizer, self.dataset.syntax_nodes)
            else:
                self.ast_types = GNN.extend_tokenizer(self.tokenizer, self.syntax_node_types)

        # ! tokenizer has to be extended before data and cache is loaded!
        # Cache graph tokenize the code accordingly!
        self.dataloaders = self.dataset.get_dataloaders()

        # set dataloaders
        self.train_dataloader = lambda: self.dataloaders["train"]
        self.val_dataloader = lambda: self.dataloaders["validation"]
        self.test_dataloader = lambda: self.dataloaders["test"]

    def setup(self, stage: str) -> None:
        self.stage = stage

    def on_train_start(self) -> None:
        # Log dataset information
        self.has_initialized = True
        try:
            self.logger.log_hyperparams(self.args)
        except Exception:
            # Fallback if log_hyperparams not supported
            pass
        self.save_hyperparameters(self.args)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["tokenizer"] = self.tokenizer
        return super().on_save_checkpoint(checkpoint)

    def configure_optimizers(self) -> Any:
        if self.has_initialized:
            return
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.gnn.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.trainer.optimizer.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.gnn.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # Initialize optimizer
        optimizer = instantiate(
            self.args.trainer.optimizer.type,
            optimizer_grouped_parameters,
            **self.args.trainer.optimizer.hyperparams,
        )

        num_update_steps_per_epoch = int(
            len(self.dataloaders["train"]) / self.args.trainer.accumulate_grad_batches
        )
        max_train_steps = self.args.trainer.epoch * num_update_steps_per_epoch

        scheduler = instantiate(
            self.args.trainer.scheduler.type,
            optimizer,
            num_warmup_steps=self.args.trainer.scheduler.num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                # "monitor": "val_loss",
                "strict": False,
                # "name": None,
            },
        }

    def training_step(self, batch, batch_idx):
        output = self.gnn(batch)
        loss = F.cross_entropy(output[batch.train_mask], batch.y[batch.train_mask])
        self.log(
            "train_loss", loss, on_epoch=True, batch_size=self.args.dataset.batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.gnn(batch)
        loss = F.cross_entropy(output[batch.val_mask], batch.y[batch.val_mask])
        self.log(
            "val_loss", loss, on_epoch=True, batch_size=self.args.dataset.batch_size
        )
        pred = torch.argmax(output, dim=1)
        correct = (pred[batch.val_mask] == batch.y[batch.val_mask]).sum().detach()
        self.log(
            "val_acc",
            correct / batch.val_mask.sum().detach(),
            on_epoch=True,
            batch_size=self.args.dataset.batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx):
        output = self.gnn(batch)
        loss = F.cross_entropy(output[batch.val_mask], batch.y[batch.val_mask])
        self.log(
            "test_loss", loss, on_epoch=True, batch_size=self.args.dataset.batch_size
        )
        pred = torch.argmax(output, dim=1)
        correct = (pred[batch.val_mask] == batch.y[batch.val_mask]).sum().detach()
        self.log(
            "test_acc",
            correct / batch.val_mask.sum().detach(),
            on_epoch=True,
            batch_size=self.args.dataset.batch_size,
        )
        return loss


def main(args):
    # Initialize the task
    task = BootstrapTask(args)
    # for bootstrap, precision is not needed
    torch.set_float32_matmul_precision("medium")
    return task
