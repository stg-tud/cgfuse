from . import Dataset
from ..preprocessors.preprocessor import all_ascii


class ConcodeDataset(Dataset):
    def __init__(self, args, tokenizer, seeder) -> None:
        super().__init__(args, tokenizer, seeder)
        self.name = "concode"
        # Set the cache path
        self.down_args["path"] = self.source
        self.cache_path = f".dataset/{self.name}_{self.pl_language}"
        self.graph_not_generated = {
            "java": {  # Code Graphs cannot be generated, therefore ignored
                "train": [16703, 25489, 54287, 69409, 79860, 92926],
                "validation": [643],
                "test": []
            },
        }

    def align_sample(self, examples):
        """This matches the gap between code representation and graph representation.
        Alignment between Graph and Text representations.

        The effect is minimal, tested on a failing case though...
        """
        for i in range(len(examples["code"])):
            # To align examples from graph preprocessor to tokenizer output. Only 4 samples in training is effected.
            examples["code"][i] = all_ascii(examples["code"][i])
            # In this dataset, strings are confescated as "srini_string"
            examples["code"][i] = examples["code"][i].replace(
                "srini_string", "str"
            )
            # Pretokenization made on these strings as whole where they correspond to different nodes in ASTs
            examples["code"][i] = examples["code"][i].replace("<<<", "< < <")
            # examples["code"][i] = examples["code"][i].replace(
            #     "<<", "< <"
            # )
            examples["code"][i] = examples["code"][i].replace(">>>", "> > >")
            examples["code"][i] = examples["code"][i].replace(">>", "> >")

            examples["nl"][i], examples["code"][i] = self.on_before_tokenize(
                examples["nl"][i], examples["code"][i]
            )
        return examples


    def postprocess(self, subset):
        # alternative, setting transform function
        # https://huggingface.co/docs/datasets/use_dataset
        self.subsets[subset] = self.subsets[subset].map(
            self.align_sample, batched=True,
        )


    def seq2seq_task(self, batch, g_batch, graphs):
        # batch = self.on_before_tokenize(batch)
        base_model_input = self.tokenize_source([b["nl"] for b in batch])

        if self.add_graphs and self.args.cache_enforce:
            grammar_tokenization, attention_mask = self.grammar_tokenization(g_batch)
            base_model_input["labels"] = grammar_tokenization
            base_model_input["decoder_attention_mask"] = attention_mask
        else:
            base_model_output = self.tokenize_target([b["code"] for b in batch])
            base_model_input["labels"] = base_model_output.input_ids
            base_model_input["decoder_attention_mask"] = (
                base_model_output.attention_mask
            )

        base_model_input = self.on_after_tokenize(base_model_input)

        return (
            # Code text encoded
            base_model_input,
            # Code Graphs
            g_batch,
        )