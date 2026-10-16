from abc import ABC
from functools import partial

import hydra
import torch
import numpy as np
from pandas import DataFrame
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
import torch_geometric.transforms as Transforms
from transformers import DataCollatorForLanguageModeling

from .batch_adapt import batch_adapt
from ..utils import get_syntax_nodes


class PdWrapperDataLoader:
    def __init__(self, pd):
        self.pd = pd

    def __len__(self):
        return len(self.pd)

    def __getitem__(self, idx):
        return self.pd.iloc[idx]


class Dataset(ABC):
    def __init__(self, args, tokenizer, seeder) -> None:
        self.args = args
        # Every dataset defines their own name
        self.name: str = ""
        self.cache_path = ""
        self.down_args = {
            "path": "",
            "name": None,
            "split": None,
            "verification_mode": "no_checks",
        }
        # depending on the type, data can be prepared in a different manner
        self.language_task = args.language_task
        # False when dataset should be prepared for pure LMing
        self.add_graphs = args.add_graphs
        # set the PL language for preprocessors
        self.pl_language = args.pl_language
        # place holder for graph generation
        self.graph_not_generated = {
            self.pl_language: {
                "train": [],
                "validation": [],
                "test": []
            },
        }

        self.cache = None
        self._tokenizer = None
        self._preprocessor = None
        self._syntax_nodes = None

        self.id_field = "id"
        self.source = args.source
        self.tokenizer = tokenizer
        self.seeder = seeder
        self.subsets = None
        self.generation_config = None

        self.start_offset = batch_adapt(self, args.model_name)
        self.cache_loaded = False

    def initialize_cache(self):
        # Do not create if subsets are not downloaded
        if getattr(self, "subsets", None) is not None:
            # Do not update if cache is already defined
            if getattr(self, "cache", None) is None:
                self.cache = {
                    n: {i: None for i in range(len(subset))}
                    for n, subset in self.subsets.items()
                }

    def load_cache(self, files_prefix):
        from ..utils import mp_load_dataset

        try:
            # Paralel load the cache and apply tokenizer on it
            self.cache = mp_load_dataset(
                files_prefix,
                self.tokenizer,
                add_edge_mask=self.preprocessor.add_edge_mask,
                process_count=self.args.num_workers,
                debug=self.args.debug,
            )
            self.cache_loaded = True

            if getattr(self, "logger", None) and getattr(
                self.logger.experiment, "log_info", None
            ):
                self.logger.experiment.log_info("load cached!")
            else:
                print("load cached!")
        except Exception:
            if getattr(self, "logger", None):
                import traceback

                if getattr(self.logger.experiment, "log_error", None):
                    self.logger.experiment.log_error(traceback.format_exc())
                else:
                    print(traceback.format_exc())

            if self.args.cache_enforce:
                raise Exception("Cache enforced but could not be loaded!")
        self.initialize_cache()

    @property
    def syntax_nodes(self):
        return None # Get all for now
        if getattr(self, "_syntax_nodes", None) is None:
            self._syntax_nodes = get_syntax_nodes(self.cache_path)
        return self._syntax_nodes

    @property
    def tokenizer(self):
        if getattr(self, "_tokenizer", None):
            return self._tokenizer
        return None

    @tokenizer.setter
    def tokenizer(self, val):
        if val is not None and val is not self.tokenizer:
            self._tokenizer = val
            # Set the tokenizer if provided
            self.preprocessor.tokenizer = val
            # Reset tokenize functions and initialize them again
            self._tokenize_source = None
            self._tokenize_target = None
            self.tokenize_source
            self.tokenize_target
            # Reset cache when tokenizer is set since it depends on
            # tokenizer output
            self.initialize_cache()

    @property
    def preprocessor(self):
        if getattr(self, "_preprocessor", None) is None:
            self._preprocessor = hydra.utils.instantiate(
                self.args.preprocessor.type, self.args.preprocessor
            )
            self._preprocessor.prepare(self)
        return self._preprocessor

    @property
    def split(self):
        # initialize if graph splitter is required (in GNN training)
        # default it Node Classification
        if getattr(self, "_split", None) is None:
            # Not implemented for this work
            if self.args.task == "NodeClassification":
                self._split = Transforms.RandomNodeSplit(
                    num_val=self.args.split_val, num_test=self.args.split_test
                )
            else:
                self._split = lambda x: x
        return self._split

    @property
    def tokenize_source(self):
        if getattr(self, "_tokenize_source", None) is None:
            self._tokenize_source = partial(
                self.tokenizer,
                return_tensors="pt",
                padding="max_length",
                max_length=self.args.max_source_length,
                truncation=True,
            )
        return self._tokenize_source

    @property
    def tokenize_target(self):
        if getattr(self, "_tokenize_target", None) is None:
            self._tokenize_target = partial(
                self.tokenizer,
                return_tensors="pt",
                padding="max_length",
                max_length=self.args.max_target_length,
                truncation=True,
            )
        return self._tokenize_target

    def code_from_sample(self, sample):
        return sample["code"]

    def prepare_sample(self, sample):
        subset_id = list(sample.keys())[-1]
        subset_name = subset_id[:-4]
        subset_idx = int(sample[subset_id])

        data = self.cache[subset_name][subset_idx]
        # If cache_enforced, data cannot be empty
        if data is None:
            data = self.preprocessor.get_pyg_data(self.code_from_sample(sample))
            # ! This wont update with multiprocessing, which is default
            self.cache[subset_name][subset_idx] = data
        # elif not self.cache_loaded:
        #     # data exists, but cache is not loaded.
        #     data = self.preprocessor.get_pyg_data_fromgraph(
        #         self.preprocessor.graph_tokenize(data, tokenizer=self.tokenizer),
        #         add_edge_mask=self.preprocessor.add_edge_mask
        #     )
        #     self.cache[subset_name][subset_idx] = data
        return data

    def max_length(self, pl_feed=None):
        if pl_feed is None:
            pl_feed = self.args.pl_feed
        if pl_feed == "input":
            max_length = self.args.max_source_length
        else:
            max_length = self.args.max_target_length
        return max_length

    def prepare_graph_batch(self, batch, pl_feed=None):
        """Adds additional pointers and values that are batch related
        which are required for the LLM / GNN alignment.
        """
        graphs = [self.prepare_sample(sample) for sample in batch]
        if pl_feed is None:
            pl_feed = self.args.pl_feed
        max_length = self.max_length(pl_feed)
        start_offset = self.start_offset[pl_feed]
        end_offset = 1
        batch = Batch.from_data_list(graphs)
        # Code ptrs
        batch.cptr = torch.zeros_like(batch.ptr)
        # Find code longer than the target length
        # +2 is there for special tokens that do not exists in GNN setup
        #   namely start token and end token. Such special tokens can be
        #   handled automatically for the alignment
        batch.extras = torch.clamp(
            batch.code_len + (start_offset + end_offset) - max_length, min=0
        )
        for i in range(len(batch.code_len)):
            batch.cptr[i + 1] = batch.cptr[i] + batch.code_len[i]
        # Decoding ptrs
        batch.gptr = batch.cptr[:-1].clone().detach()
        # Use cptr as end points which are clamped to fit max target length
        # batch.cptr[1:] = batch.cptr[1:] - batch.extras

        return batch, graphs

    def grammar_tokenization(self, graphs, pl_feed=None):
        """
        This only cares for bos, eos, and padding and create a representation from
        code. What if code representation also contains text elements? Like CodeT5
        starting with "<java>: " indicator?
        """
        if pl_feed is None:
            pl_feed = self.args.pl_feed
        max_length = self.max_length(pl_feed)
        start_offset = self.start_offset[pl_feed]
        pad_token_id = torch.tensor(self.generation_config.pad_token_id)
        # These are in array form since they will be used for concat

        # This work for the target 9 models but by chance
        # It is better to use self.args.model_name and
        # set all needed elsewhere per model, etc. config or main task
        if start_offset == 1:
            bos_token_id = torch.tensor(
                [
                    (
                        self.generation_config.bos_token_id
                        if pl_feed == "input"
                        else self.generation_config.decoder_start_token_id
                    )
                ]
            )
            # TODO: This is a test for CodeT5, since its bos is 0, which is also pad
            if self.args.model_name in ["codet5", "t5"]:
                bos_token_id = self.generation_config.bos_token_id
        eos_token_id = torch.tensor([self.generation_config.eos_token_id])

        # Strip grammar tokens and split by sample
        splits = torch.split(graphs.x[graphs.code_mask], graphs.code_len.tolist())
        # This precut is for memory efficiency to apply before pad_sequence
        splits = [s[:max_length] if len(s) > max_length else s for s in splits]
        # Add end of the sentence token
        splits = [torch.cat((s, eos_token_id), 0) for s in splits]
        # Create padded tensor from samples
        recreated = pad_sequence(splits, True, pad_token_id)
        # Create padded tensor for batch
        output_ids = torch.tile(
            pad_token_id,
            (
                # self.args.batch_size if not self.args.debug else 4,
                graphs.code_len.size(0),# if not self.args.debug else 4,
                max_length,
            ),
        )
        # set starting sequence (now only have bos)
        if start_offset == 1:  # Dependent on the model and feed direction
            output_ids[..., 0] = bos_token_id
        # get truncation
        clamp = recreated.shape[1] + start_offset
        clamp = min(int(clamp), max_length)
        # set graph tensor
        output_ids[..., start_offset : clamp] = recreated[
            ..., : clamp - start_offset
        ]
        # Add end of sentence tokens if truncated
        trunc_change = output_ids[..., -1] != self.generation_config.pad_token_id
        output_ids[..., -1][trunc_change] = eos_token_id

        attention_mask = output_ids != self.generation_config.pad_token_id

        return output_ids, attention_mask.to(torch.int64)

    def postprocess(self, subset):
        """ called after dataset and cache are downloaded
        and corrections are applied, per subset.
        Followed by
        """
        pass

    def dataset_corrections(self):
        if not self.args.debug:
            # Delete problematic samples
            for subset in self.graph_not_generated[self.pl_language]:
                exclude_idx = self.graph_not_generated[self.pl_language][subset]
                if not exclude_idx:
                    continue

                for _id in exclude_idx:
                    del self.cache[subset][_id]

                self.subsets[subset] = self.subsets[subset].select(
                    (
                        i
                        for i in range(len(self.subsets[subset]))
                        if i not in set(exclude_idx)
                    )
                )

            # Check if cache and dataset entries are the same!
            for subs_name, subset in self.subsets.items():
                ds_len = len(self.subsets[subs_name])
                ch_len = len(self.cache[subs_name])
                assert (
                    ds_len == ch_len
                ), f"{subs_name} are lengths do not match, {ds_len}, {ch_len}"

                diff = set(subset[self.id_field]).difference(set(self.cache[subs_name].keys()))
                assert len(diff) == 0, f"{subs_name} has different entries: {diff}"

    def load_dataset(self):
        # do not load again if already loaded!
        if self.subsets:
            return

        self.subsets = load_dataset(**self.down_args)
        self.load_cache(self.cache_path)

        # Add ids if not exists for following operations
        for subset in self.subsets:
            if self.id_field not in self.subsets[subset].features:
                self.subsets[subset] = self.subsets[subset].add_column(
                    self.id_field, list(range(len(self.subsets[subset])))
                )

        self.dataset_corrections()

        # Fix ids if not exists for following operations
        for subset in self.subsets:
            new_id = f"{subset}_idx"
            self.postprocess(subset)

            # E.g. CodeSearchNet already have an id field
            # since corrections may remove samples, we should use
            # defined ids rather than defining ourselves.
            self.subsets[subset] = self.subsets[subset].add_column(
                new_id, self.subsets[subset][self.id_field]
            )
            # Relocate id column to the end for all datasets for easy matching
            new_order = [c for c in self.subsets[subset].column_names if c != new_id] + [new_id]
            self.subsets[subset] = self.subsets[subset].select_columns(new_order)

    def get_dataloaders(self):
        if not self.subsets:
            self.load_dataset()
        return {
            subset: self.get_dataloader(self.subsets[subset]) for subset in self.subsets
        }

    def get_dataloader(self, subset) -> DataLoader:
        if isinstance(subset, DataFrame):
            subset = PdWrapperDataLoader(
                subset.head(128) if self.args.debug else subset
            )
        else:
            subset = subset.select(range(128)) if self.args.debug else subset
        return DataLoader(
            subset,
            batch_size=self.args.batch_size if not self.args.debug else 32,
            # Only shuffle when training and not debug
            shuffle=subset == "train" and not self.args.debug,
            generator=self.seeder.torch_gen(),
            persistent_workers=not self.args.debug,
            num_workers=(0 if self.args.debug else self.args.num_workers),
            collate_fn=self._collate_fn,
        )

    def classification_task(self, batch):
        # Encode input using model's tokenizer
        batch_enc = self.tokenize_target(
            [self.code_from_sample(sample) for sample in batch]
        )
        # Mask the input text
        trans_masked_input = self.mlm_masker([b for b in batch_enc.input_ids])

        base_model_input = {
            "attention_mask": batch_enc["attention_mask"],
            # input ids are "mask" for 80% of the masked tokens, are random for
            # 10% of the masked tokens, and are the unchanged for the 10% remaining
            "input_ids": trans_masked_input["input_ids"],
            # labels are -100 for all non-masked tokens and are the value to predict
            # for the masked token (same as input_ids)
            "labels": trans_masked_input["labels"],
        }

        if not self.add_graphs:
            return (base_model_input, None)

        # mask trees according to labels in trans_masked_input
        # ! Check here for alignment later
        # get attention masks
        att_masks = np.array(batch_enc["attention_mask"], dtype=np.bool)
        # get labels of tokens, cut <s> and </s> tokens
        filtered_labels = [
            trans_masked_input["labels"][i][att_masks[i]][1:-1]
            for i in range(len(batch))
        ]

        return (
            # Code text encoded
            base_model_input,
            # Code Graphs
            Batch.from_data_list(
                [
                    self.tree_masker(self.prepare_sample(sample), filtered_labels[i])
                    for i, sample in enumerate(batch)
                ]
            ),
        )

    def seq2seq_task(self, batch, g_batch, graphs):
        pass

    def graph_task(self, batch):
        return Batch.from_data_list(
            [self.split(self.prepare_sample(sample)) for sample in batch]
        )

    def _collate_fn(self, batch):
        if self.language_task == "classification":
            return self.classification_task(batch)
        elif self.language_task == "seq2seq":
            g_batch, graphs = None, None
            if self.add_graphs:
                g_batch, graphs = self.prepare_graph_batch(batch)

            return self.seq2seq_task(batch, g_batch, graphs)
        else:
            return self.graph_task(batch)
