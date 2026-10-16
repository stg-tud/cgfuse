from typing import Any, Dict
import re, logging

import lightning.pytorch as pl
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from aim import Text

from transformers.trainer_pt_utils import get_parameter_names
import adapters
import torch

from .task import Task, TaskStage, EmbSource
from ..utils import GeometricAdapterConfig, Seeder
from ..utils.adapter import add_adapter
from ..models import GNN


class NoParsingFilter(logging.Filter):
    def filter(self, record):
        # This is handled manually but causes too much log pollution.
        return (
            not record.getMessage()
            == "There are adapters available but none are activated for the forward pass."
        )


logger = logging.getLogger("adapters.model_mixin")
logger.addFilter(NoParsingFilter())


class StructInfuseTask(Task, pl.LightningModule):
    def __init__(self, args: DictConfig):
        Task.__init__(self, args)
        pl.LightningModule.__init__(self)

        self.automatic_optimization = self.args.trainer.automatic_optimization

        self.configure_metrics()
        self.train_reported = True

    def initialize_dataset(self, dataset_config, tokenizer=None):
        dataset_config.language_task = self.args.language_task
        dataset_config.debug = self.args.debug
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

        self.tokenizer = None
        if self.args.trainer.fusion.teacher:
            # load teacher early to access tokenizer to set to dataset
            expert_path = self.args.trainer.fusion.expert
            if expert_path is None:
                raise ValueError(
                    "Set trainer.fusion.expert=<path to a GNN checkpoint from task=bs_gnn_jv>"
                )
            self.checkpoint = torch.load(
                expert_path,
                map_location=lambda storage, loc: storage,  # Autocheckpoint also saves the device. This is to fix that
            )
            self.tokenizer = self.checkpoint["tokenizer"]

        dataset_config = self.args.dataset
        self.test_dataloader_names = []
        self.train_dataloader_name = ""
        self.train_dataloader_name = dataset_config.source
        self.test_dataloader_names.append(dataset_config.source)
        self.dataset = self.initialize_dataset(dataset_config, self.tokenizer)
        self.start_offset = self.dataset.start_offset

        self.dataset.load_dataset()

        # Set dataloaders after `self.configure_model` when generation_config is prepared!
        self.test_outputs.append([])
        self.test_gold.append([])

    def on_train_start(self) -> None:
        # This comes after prepare_data, setup, configure_model, configure_optimizers
        self.has_initialized = True
        if not self.args.debug and hasattr(self.logger, "experiment"):
            # Use Lightning's hyperparameter logging which works across loggers
            try:
                # Log hyperparameters calculated during setup
                self.args.dataset.start_offset = self.dataset.start_offset
                if hasattr(self.adapters[-1], "start_offset_func"):
                    self.args.dataset.start_offset_func = self.adapters[-1].start_offset_func
                self.logger.log_hyperparams(self.args)
            except Exception:
                # Fallback if log_hyperparams not supported
                pass
        self.save_hyperparameters(self.args)

    def load_teacher(self) -> GNN:
        gnn_params = self.checkpoint["hyper_parameters"]["gnn"]

        gnn_params["vocab_size"] = len(self.tokenizer)
        gnn: GNN = instantiate(gnn_params["type"], gnn_params)
        if len(self.tokenizer) > gnn.embeddings.weight.size(0):
            gnn.embeddings = gnn.resize_token_embeddings(len(self.tokenizer))
        gnn.load_state_dict(
            {
                key[4:] if key.startswith("gnn") else key: val
                for key, val in self.checkpoint["state_dict"].items()
            }
        )

        del self.checkpoint
        return gnn

    def shared_params_adapters(self):
        if getattr(self.model, "shared_parameters", None) is None:
            shared_parameters = self.model.base_model.shared_parameters[
                self.attacment_name
            ]
        else:
            shared_parameters = self.model.shared_parameters[self.attacment_name]

        return shared_parameters

    def get_syntax_embeddings_generator(self, gnn_embeddings):
        if self.args.dataset.model_name == "t5":
            # Original t5 have extra empty 28 weights in its input embeddings
            self.old_num_tokens = 32100

        # Extract learned syntax token embedding matrix
        syntax_token_embeddings_data = gnn_embeddings.weight.data[
            self.old_num_tokens :, :
        ]

        # Generate embeddings per GNN when initialized
        def create_syntax_token_embeddings():
            syntax_token_embeddings = torch.nn.Embedding(
                syntax_token_embeddings_data.size()[0],
                syntax_token_embeddings_data.size()[1],
            )
            syntax_token_embeddings.weight.data = (
                syntax_token_embeddings_data.detach().clone()
            )
            return syntax_token_embeddings

        return create_syntax_token_embeddings

    def configure_embeddings(self, gnn_embeddings):
        # Get sizes
        # e.g. 32835
        new_num_tokens = len(self.tokenizer)
        # e.g. 32100, 768
        self.old_num_tokens, old_embedding_dim = (
            self.model.get_input_embeddings().weight.size()
        )

        create_syntax_token_embeddings = self.get_syntax_embeddings_generator(
            gnn_embeddings
        )

        new_embeddings = self.model.get_input_embeddings()
        self.standardize_embeddings(new_embeddings)

        return create_syntax_token_embeddings

    def configure_model(self):
        if self.has_initialized:
            return

        # Initialize model
        self.model = call(self.args.model.type, self.args.model.hf_handle)

        if self.args.trainer.fusion.teacher:
            # Load GNN
            gnn = self.load_teacher()

            # Strip GNN to get raw form, save GNN embeddings
            gnn_wo, gnn_embeddings = gnn.strip()

            if self.emb_source.require_merge:
                create_syntax_token_embeddings = self.configure_embeddings(
                    gnn_embeddings
                )

                pad_emb = self.model.get_input_embeddings().weight[
                    self.tokenizer.pad_token_id, :
                ]

            # Prepare model for teacher
            adapters.init(self.model)

            self.attacment_name = "gnn_adapter"

            # Prepare for shared embeddings for teacher.
            parameters = torch.nn.ParameterDict()
            if getattr(self.model, "shared_parameters", None) is None:
                self.model.base_model.shared_parameters[self.attacment_name] = (
                    parameters
                )
            else:
                self.model.shared_parameters[self.attacment_name] = parameters

            # Set learned GNN embeddings as shared among all GNNs to use
            if self.emb_source == EmbSource.SELF:
                # Make embeddings shared.
                parameters["gnn_embeddings"] = gnn_embeddings
                pad_emb = gnn_embeddings.weight[self.tokenizer.pad_token_id, :]

            parameters["pad_emb"] = pad_emb

            # Initialize adapter config
            adapter_config = GeometricAdapterConfig(
                scaling=self.args.trainer.fusion.scale,
            )

            self.adapters = []
            # codet5, t5, bart, plbart sends full body.
            m_check = any(idf in self.args.model.name for idf in ["bart", "t5"])
            fusion_cfg = self.args.trainer.fusion
            target_module = self.model if m_check else self.model.decoder
            # Layer-wise addition of teacher model
            out_adapters = add_adapter(
                target_module,
                self.attacment_name,
                adapter_config,
                fusion_cfg.target_layers,
                gnn,
                is_encoder=False,
                # Extra parameters to directly set to every adapter
                syntax_token_embeddings=(
                    create_syntax_token_embeddings()
                    if self.emb_source == EmbSource.LAYER
                    else None
                ),
                start_offset=self.dataset.start_offset,
                model_name=self.args.dataset.model_name
            )

            for adapter in out_adapters:
                adapter.emb_source = self.emb_source.value
                adapter.start_offset = self.start_offset
                adapter.pl_feed = self.args.dataset.pl_feed
                self.adapters.append(adapter)

            self.adapters[-1].adv_ptr = True
            self.model.set_active_adapters(self.attacment_name)

            # Remove unnecessary modules
            del gnn_wo, gnn_embeddings
        else:
            # Fallback to default tokenizer
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
            self.dataset.tokenizer = self.tokenizer

        self.configure_special_tokens()

        # Load and get dataloader
        # Loading is delayed until the model is fully configured
        # in case there is a cache loading where entire dataset is preprocessed with
        # corresponding tokenizer with set special tokens
        self.dataloaders = self.dataset.get_dataloaders()
        # set dataloaders
        self.train_dataloader = lambda: self.dataloaders["train"]
        self.val_dataloader = lambda: self.dataloaders["validation"]
        self.test_dataloader = lambda: self.dataloaders["test"]

        if not self.args.trainer.automatic_optimization:
            torch.set_grad_enabled(True)

    def configure_optimizers(self) -> Any:
        if self.has_initialized:
            return
        # Parameter names without LayerNorm.weight
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        # Parameter names without bias
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        # Find adapter parameters among decay parameters
        if hasattr(self.model, "config") and hasattr(self.model.config, "adapters"):
            match_str = r"adapter_fusion_layer\..*\.value"
            decay_parameters = [
                name for name in decay_parameters if not re.match(match_str, name)
            ]
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

        window = (
            self.args.trainer.fusion.adapting_epoch + self.args.trainer.fusion.epoch
        )

        scheduler = instantiate(
            self.args.trainer.scheduler.type,
            optimizer,
            num_warmup_steps=self.args.trainer.scheduler.num_warmup_steps,
            num_training_steps=self.get_train_steps(
                (self.args.trainer.epoch / window) * self.args.trainer.fusion.epoch
            ),
        )

        if not self.args.trainer.automatic_optimization:
            # Always start with adapting if experts are provided
            self.task_stage = TaskStage.ADAPTING
            # Set model parameters freeze and requires_grad false
            self.model.train_adapter(self.attacment_name)
            # Get the rest of the parameters, GNN parameters
            opt_grouped_teacher_params = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.trainer.fusion.optimizer.weight_decay,
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

            # Initialize teacher optimizer and scheduler
            optimizer_teacher = instantiate(
                self.args.trainer.fusion.optimizer.type,
                opt_grouped_teacher_params,
                **self.args.trainer.fusion.optimizer.hyperparams,
            )
            scheduler_teacher = instantiate(
                self.args.trainer.fusion.scheduler.type,
                optimizer_teacher,
                num_warmup_steps=self.args.trainer.fusion.scheduler.num_warmup_steps,
                num_training_steps=self.get_train_steps(
                    (self.args.trainer.epoch / window)
                    * self.args.trainer.fusion.adapting_epoch
                ),
            )

            return ([optimizer_teacher, optimizer], [scheduler_teacher, scheduler])

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

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["tokenizer"] = self.tokenizer
        return super().on_save_checkpoint(checkpoint)

    def on_embeddings_called(self, input_ids):
        if len(input_ids.shape) == 1: # GNN embeddings are generated
            return

        if input_ids.shape[1] > 1:
            # When full input is sent, this could be encoder side
            # or decoder side when training
            self.adapters[-1].input_ids = input_ids
        else:
            if self.adapters[-1].cursor == 0:
                self.adapters[-1].input_ids = input_ids
            else:
                self.adapters[-1].input_ids = torch.cat([self.adapters[-1].input_ids, input_ids], dim=-1)

    def set_batch_gnn(self, batch, batch_idx, is_train=False):
        if not self.args.trainer.fusion.teacher:
            return
        # Gets embeddings for the graph batch and sets it for adapters to access
        # Get latent representations for GNN from its embeddings
        shared_parameters = self.shared_params_adapters()
        embeddings = (
            self.model.get_input_embeddings()
            if self.emb_source.require_merge
            else shared_parameters["gnn_embeddings"]
        )

        # Update shared batch for GNNs to access
        if self.args.debug:
            shared_parameters["transformer_batch"] = batch[0]
        self.last_input_ids = batch[0]["input_ids" if self.args.dataset.pl_feed == "input" else "labels"]
        shared_parameters["graph_batch"] = batch[1]
        # If teachers are fed with layer features
        if self.emb_source == EmbSource.LAYER:
            for adapter in self.adapters:
                adapter.is_train = is_train
                adapter.code_indices = batch[1].code_mask.nonzero().squeeze()

                # Set embeddings to every adapter and reset their graph cursor
                syntax_token_embeddings = getattr(
                    adapter, "syntax_token_embeddings", None
                )

                if syntax_token_embeddings is not None:  # and False, for debugging
                    adapter.past_values = torch.empty(
                        len(batch[1].x),
                        syntax_token_embeddings.weight.size(1),
                        device=batch[1].x.device,
                    )
                    syntax_indices = (~batch[1].code_mask).nonzero().squeeze()
                    # Get input IDs for syntax tokens, assuming input ids are added to the end
                    syntax_inp_ids = batch[1].x[syntax_indices] - self.old_num_tokens
                    adapter.past_values[syntax_indices] = syntax_token_embeddings(
                        syntax_inp_ids
                    )

                    adapter.past_values[adapter.code_indices] = (
                        embeddings(batch[1].x[adapter.code_indices]).detach().clone()
                    )
                else:
                    graph_embeddings = embeddings(batch[1].x)
                    adapter.past_values = graph_embeddings.detach().clone()

                if is_train:
                    adapter.cursor = self.args.dataset.max_target_length
                else:
                    adapter.cursor = 0

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.set_batch_gnn(batch, batch_idx, is_train=True)

    def on_validation_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ):
        self.set_batch_gnn(batch, batch_idx)

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self.set_batch_gnn(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()[self.task_stage.i - 1]
        sch = self.lr_schedulers()[self.task_stage.i - 1]

        is_last_batch_to_accumulate = (
            (batch_idx + 1) % self.args.trainer.accumulate_grad_batches == 0
            or self.trainer.is_last_batch
        )

        with opt.toggle_model(sync_grad=is_last_batch_to_accumulate):
            outputs = self.model(**batch[0])

            # ? This breaks when accumulation is set and grads are not used between batches?
            opt.zero_grad()
            self.manual_backward(outputs.loss)

            if is_last_batch_to_accumulate:
                if self.task_stage != TaskStage.ADAPTING:
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

        self.log_dict(
            {
                "train_loss": outputs.loss.detach(),
            },
            on_epoch=True,
            batch_size=batch_size,
            logger=True,
        )

        return outputs.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        is_sanity = self.trainer.state.stage.title() == "Sanity_Check"
        if is_sanity and self.args.debug:
            self.batch = batch[0]
        max_target_length = self.args.dataset.max_target_length

        generated_tokens = self.model.generate(
            batch[0]["input_ids"],
            attention_mask=batch[0]["attention_mask"],
            max_length=max_target_length,
            return_dict_in_generate=is_sanity and self.args.debug,
            num_beams=1,
        )
        if is_sanity and self.args.debug:
            print(f"Generate first token: {generated_tokens.sequences[0][0]}")
            generated_tokens = generated_tokens.sequences

        # Get label text
        labels = self.tokenizer.batch_decode(
            batch[0]["labels"].cpu().numpy(), skip_special_tokens=True
        )

        preds = self.tokenizer.batch_decode(
            generated_tokens.cpu().numpy(), skip_special_tokens=True
        )

        self.validation_outputs[dataloader_idx].extend(preds)
        self.validation_gold[dataloader_idx].extend(labels)

        if len(self.validation_outputs_special_tokens[dataloader_idx]) < 10:
            st_labels = self.tokenizer.batch_decode(
                batch[0]["labels"].cpu().numpy(), skip_special_tokens=False
            )
            self.validation_gold_special_tokens[dataloader_idx].extend(st_labels)

            st_preds = self.tokenizer.batch_decode(
                generated_tokens.cpu().numpy(), skip_special_tokens=False
            )
            self.validation_outputs_special_tokens[dataloader_idx].extend(st_preds)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if not self.args.trainer.automatic_optimization:
            self.eval()

        max_target_length = self.args.dataset.max_target_length
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

        self.test_outputs[dataloader_idx].extend(preds)
        self.test_gold[dataloader_idx].extend(labels)

    def on_test_epoch_start(self):
        # After reload (to best checkpoint), these values were not set...
        for adapter in self.adapters:
            adapter.emb_source = self.emb_source.value
            adapter.start_offset = self.dataset.start_offset

    def on_train_epoch_start(self):
        self.train_reported = False
        # Alternating training if there is a teacher
        # between GNN training and full training.
        if self.args.trainer.fusion.teacher:
            # Activate the adapters, in case they are disabled bc of the beam search
            self.model.set_active_adapters(self.attacment_name)
            # always start with adapting and train main model for fusion.epoch times
            # then adapt again and repeat
            window = self.trainer.current_epoch % (
                self.args.trainer.fusion.adapting_epoch + self.args.trainer.fusion.epoch
            )
            if window < self.args.trainer.fusion.adapting_epoch:
                # Start with adapting stage in the window
                self.task_stage = TaskStage.ADAPTING
                # Freeze all other parameters but GNNs while adapting GNNs to layers
                self.model.train_adapter(self.attacment_name)
            else:
                # Continue with finetuning stage when experts are adapted
                self.task_stage = TaskStage.FINETUNING
                # unfreeze model for finetuning after adaptation
                self.model.freeze_model(False)
                # unfreeze model and freeze GNNs, initial tests show ~-1% difference
                # freeze_adapters(self.model, self.attacment_name)

    def track_adapter_inputs(self, subset):
        """Logs gold vs. adapter-seen tokens of 5 samples to Aim (skipped for other loggers)."""
        if not hasattr(getattr(self.logger, "experiment", None), "track"):
            return
        gold = self.tokenizer.batch_decode(
            self.last_input_ids.cpu().numpy()[:5], skip_special_tokens=False
        )
        report = self.tokenizer.batch_decode(
            self.adapters[-1].input_ids.cpu().numpy()[:5], skip_special_tokens=False
        )
        self.logger.experiment.track(
            Text("\n".join([f"{g}\n{r}\n" for g, r in zip(gold, report)])),
            name="adapter",
            context={"subset": subset},
            epoch=self.current_epoch,
        )

    def on_validation_epoch_start(self) -> None:
        # This is for reporting training examples also
        # since training epoch end called after validation epoch ends
        if self.train_reported:
            return super().on_validation_epoch_start()

        self.track_adapter_inputs("train")
        self.train_reported = True
        self.adapters[-1].input_ids = None

    def on_validation_epoch_end(self) -> None:
        # Only validation and clearing
        self.evaluate(self.validation_outputs, self.validation_gold)
        self.validation_outputs.clear()
        self.validation_outputs.append([])
        self.validation_gold.clear()
        self.validation_gold.append([])
        self.validation_outputs_special_tokens.clear()
        self.validation_outputs_special_tokens.append([])
        self.validation_gold_special_tokens.clear()
        self.validation_gold_special_tokens.append([])

        # For debugging purposes, log some examples from validation
        if self.trainer.state.stage.title() != "Sanity_Check":
            self.track_adapter_inputs("validation")
        self.adapters[-1].input_ids = None

    def on_test_epoch_end(self) -> None:
        # Only validation and clearing
        self.evaluate(self.test_outputs, self.test_gold, "all")
        self.test_outputs.clear()
        self.test_gold.clear()
        for _ in self.test_dataloader_names:
            self.test_outputs.append([])
            self.test_gold.append([])
        self.adapters[-1].input_ids = None


def main(args):
    # Initialize the task
    task = StructInfuseTask(args)
    torch.set_float32_matmul_precision("medium")  # "medium", "high"
    return task
