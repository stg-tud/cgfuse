from typing import Any, Dict
from omegaconf import DictConfig
from hydra.utils import call, instantiate
import lightning.pytorch as pl

import torch
from transformers.trainer_pt_utils import get_parameter_names

from .task import Task
from ..utils import Seeder

class BaselineTask(Task, pl.LightningModule):
    def __init__(self, args: DictConfig):
        Task.__init__(self, args)
        pl.LightningModule.__init__(self)

        self.automatic_optimization = self.args.trainer.automatic_optimization

        self.configure_metrics()

    def initialize_dataset(self, dataset_config, tokenizer=None):
        dataset_config.language_task = self.args.language_task
        dataset_config.debug = self.args.debug
        # If cache enforce is True, this is needed in datasets to
        # generate representations from graphs rather than text
        # namely: graph tokenization.
        dataset_config.add_graphs = True
        dataset_config.model_name = self.args.model.name

        # Initialize dataset
        # Tokenizer set to none here since it is coming from model configuration
        dataset = instantiate(
            dataset_config.type, dataset_config, None, Seeder(self.args.seed)
        )
        dataset.logger = self.logger
        if tokenizer is not None:
            dataset.tokenizer = tokenizer
        return dataset

    def prepare_data(self) -> None:
        if self.has_initialized or getattr(self, "data_prepared", False):
            return
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

        dataset_config = self.args.dataset
        self.test_dataloader_names = []
        self.train_dataloader_name = ""
        self.train_dataloader_name = dataset_config.source
        self.test_dataloader_names.append(dataset_config.source)
        self.dataset = self.initialize_dataset(dataset_config, self.tokenizer)
        self.start_offset = self.dataset.start_offset

        self.dataset.load_dataset()

        self.test_outputs.append([])
        self.test_beam_outputs.append([])
        self.test_gold.append([])

    def on_train_start(self) -> None:
        # This comes after prepare_data, setup, configure_model, configure_optimizers
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

    def configure_model(self):
        if self.has_initialized:
            return

        self.dataset.tokenizer = self.tokenizer
        self.model = call(self.args.model.type, self.args.model.hf_handle)

        self.configure_special_tokens()
        self.standardize_embeddings(self.model.get_input_embeddings())

        self.dataloaders = self.dataset.get_dataloaders()

        # set dataloaders
        self.train_dataloader = lambda: self.dataloaders["train"]
        self.val_dataloader = lambda: self.dataloaders["validation"]
        self.test_dataloader = lambda: self.dataloaders["test"]

    def configure_optimizers(self) -> Any:
        if self.has_initialized:
            return
        # Parameter names without LayerNorm.weight
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        # Parameter names without bias
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        opt_grouped_all_params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.trainer.optimizer.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        # Initialize optimizer
        optimizer = instantiate(
            self.args.trainer.optimizer.type,
            opt_grouped_all_params,
            **self.args.trainer.optimizer.hyperparams,
        )

        scheduler = instantiate(
            self.args.trainer.scheduler.type,
            optimizer,
            num_warmup_steps=self.args.trainer.scheduler.num_warmup_steps,
            num_training_steps=self.get_train_steps(),
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
        if self.args.trainer.automatic_optimization:
            outputs = self.model(**batch[0])
        else:
            opt = self.optimizers()
            sch = self.lr_schedulers()

            is_last_batch_to_accumulate = (
                (batch_idx + 1) % self.args.trainer.accumulate_grad_batches == 0
                or self.trainer.is_last_batch
            )

            with opt.toggle_model(sync_grad=is_last_batch_to_accumulate):
                outputs = self.model(**batch[0])

                opt.zero_grad()
                self.manual_backward(outputs.loss)

                if is_last_batch_to_accumulate:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.args.trainer.gradient_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                    opt.step()
                    # opt.zero_grad()
                    sch.step()

        batch_size = (
            self.args.dataset.train.batch_size
            if "train" in self.args.dataset
            else self.args.dataset.batch_size
        )

        self.log(
            "train_loss",
            outputs.loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
            logger=True,
        )

        return outputs.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        max_target_length = self.args.dataset.max_target_length

        generated_tokens = self.model.generate(
            batch[0]["input_ids"],
            attention_mask=batch[0]["attention_mask"],
            max_length=max_target_length
        )

        # Get label text
        labels = self.tokenizer.batch_decode(
            batch[0]["labels"].cpu().numpy(), skip_special_tokens=True
        )

        preds = self.tokenizer.batch_decode(
            generated_tokens.cpu().numpy(), skip_special_tokens=True
        )

        self.validation_outputs[dataloader_idx].extend(preds)
        self.validation_gold[dataloader_idx].extend(labels)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if not self.args.trainer.automatic_optimization:
            self.eval()

        max_target_length = self.args.dataset.max_target_length

        beam_generated_tokens = self.model.generate(
            batch[0]["input_ids"],
            attention_mask=batch[0]["attention_mask"],
            max_length=max_target_length,
            num_beams=self.args.trainer.generation.beam_size,
            early_stopping=True,
        )

        generated_tokens = self.model.generate(
            batch[0]["input_ids"],
            attention_mask=batch[0]["attention_mask"],
            max_length=max_target_length,
            num_beams=1,
        )

        # Get label text
        labels = self.tokenizer.batch_decode(
            batch[0]["labels"].cpu().numpy(), skip_special_tokens=True
        )

        preds = self.tokenizer.batch_decode(
            generated_tokens.cpu().numpy(), skip_special_tokens=True
        )

        beam_preds = self.tokenizer.batch_decode(
            beam_generated_tokens.cpu().numpy(), skip_special_tokens=True
        )

        self.test_outputs[dataloader_idx].extend(preds)
        self.test_beam_outputs[dataloader_idx].extend(beam_preds)
        self.test_gold[dataloader_idx].extend(labels)

    def on_validation_epoch_end(self) -> None:
        # Only validation and clearing
        self.evaluate(self.validation_outputs, self.validation_gold)
        self.validation_outputs.clear()
        self.validation_outputs.append([])
        self.validation_beam_outputs.clear()
        self.validation_beam_outputs.append([])
        self.validation_gold.clear()
        self.validation_gold.append([])

    def on_test_epoch_end(self) -> None:
        # Only validation and clearing
        self.evaluate(self.test_outputs, self.test_gold, "all")
        if len(self.test_beam_outputs[0]) != 0:
            self.evaluate(self.test_beam_outputs, self.test_gold, "all", is_beam=True)
        self.test_outputs.clear()
        self.test_beam_outputs.clear()
        self.test_gold.clear()

def main(args):
    # Initialize the task
    task = BaselineTask(args)
    torch.set_float32_matmul_precision("medium")  # "medium", "high"
    return task
