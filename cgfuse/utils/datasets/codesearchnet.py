import torch
from transformers import BatchEncoding
from . import Dataset
from ..preprocessors.preprocessor import all_ascii


# Each would define their own preprocessor
class CodeSearchNetDataset(Dataset):
    def __init__(self, args, tokenizer, seeder) -> None:
        super().__init__(args, tokenizer, seeder)
        self.name = "codesearchnet"
        self.cache_path = f".dataset/{self.name}_{self.pl_language}"
        self.down_args.update({
            "path": self.source,
            "name": self.pl_language,
        })
        self.graph_not_generated = {
            "java": { # CodeGraphs cannot be generated, therefore ignored
                "train": [108926, 109283],
                "validation": [],
                "test": []
            },
            "python": { # CodeGraphs cannot be generated, therefore ignored
                "train": [2861, 6436, 9228, 36058, 40437, 52237, 53542, 103236, 103242, 103257, 103272, 103286, 103534, 158767, 158772, 189490, 220635, 223095, 223096],
                "validation": [],
                "test": []
            },
            "javascript": { # CodeGraphs cannot be generated, therefore ignored
                "train": [17614, 17615, 26116, 31279, 35143, 40855, 42381, 44262, 54689],
                "validation": [],
                "test": []
            },
            "go": { # CodeGraphs cannot be generated, therefore ignored
                "train": [8369, 16256, 16257, 82733, 140109, 151765],
                "validation": [],
                "test": []
            },
            "ruby": { # CodeGraphs cannot be generated, therefore ignored
                "train": [],
                "validation": [],
                "test": []
            },
            "php": { # CodeGraphs cannot be generated, therefore ignored
                "train": [1234, 2402, 2746, 5367, 5467, 5547, 6964, 8552, 8553, 8554, 8555, 8556, 18279, 19277, 19485, 20268, 20738, 26815, 31052, 31345, 31346, 33422, 33424, 33566, 33657, 35418, 35616, 36178, 36549, 39900, 42229, 42231, 44818, 46402, 47441, 49693, 49696, 49708, 49715, 50406, 51065, 53963, 57814, 61063, 61066, 61307, 61308, 61312, 61397, 62212, 62292, 63801, 64956, 64959, 65724, 66877, 67088, 67517, 67520, 67521, 68558, 69056, 69083, 69431, 70417, 71816, 71818, 71820, 71822, 73811, 76500, 76501, 76569, 77552, 77784, 78155, 79982, 81503, 81504, 81528, 81536, 81604, 81743, 81809, 81810, 81811, 88257, 88260, 88262, 88266, 88280, 88305, 88575, 88579, 90867, 91451, 92166, 92167, 94121, 94146, 94149, 94150, 96549, 97330, 97795, 97796, 97797, 98764, 100771, 101346, 101774, 102272, 103629, 103793, 104009, 104872, 104873, 104886, 106151, 106153, 106869, 107018, 109347, 110764, 112940, 114918, 116258, 117902, 118753, 119166, 119351, 120320, 122551, 124254, 124264, 126130, 126172, 126180, 126190, 126275, 126286, 126300, 126394, 126416, 126515, 126548, 126570, 126628, 126629, 126643, 126830, 131020, 131496, 132374, 135844, 138044, 138152, 138172, 138173, 143509, 143656, 143657, 143658, 143659, 146330, 146333, 146553, 146640, 146779, 148796, 149041, 150239, 152077, 153339, 153944, 154493, 154504, 157368, 164878, 165296, 165403, 165663, 165836, 165964, 166492, 167801, 169608, 170909, 171460, 172546, 173491, 173492, 173723, 174240, 174581, 174795, 175106, 175777, 175778, 175821, 176106, 177562, 180263, 180796, 180829, 183639, 183674, 183766, 183873, 183874, 183879, 184529, 184665, 185109, 185110, 187466, 188067, 190778, 191214, 191622, 191677, 191678, 192413, 193334, 196049, 198372, 198374, 198376, 200129, 200140, 202157, 202664, 203920, 205896, 212404, 212405, 212408, 212409, 212410, 212411, 212413, 212414, 212415, 213376, 214257, 219007, 220480, 220528, 223540, 223541, 223545, 225569, 226164, 226734, 226788, 229466, 230736, 230801, 230829, 235342, 236027, 237139, 237146, 237570, 238295, 238619, 239428, 239969, 240155],
                "validation": [771, 1311, 5873, 5918, 6006, 10331, 11039, 11041, 12541],
                "test": [277, 278, 1799, 3820, 6721, 8332, 8333, 10998, 11720, 12097, 12546, 12734, 12990, 12991, 12996, 13440]
            },
        }

        self.stop_words = ["'''", '"""'] if self.pl_language == "python" else ["\n"]

    @staticmethod
    def align_sample(examples):
        def remove_comments(code_tokens):
            # We assume tokenization already removed the multiline comments or at least crushed them into single line
            comment_starters = [
                "#",        # Python, Ruby, PHP (single-line)
                "//",       # Go, JavaScript, PHP, Java (single-line)
                "/*",       # Go, JavaScript, PHP, Java (multi-line start)
                # "*/",       # Go, JavaScript, PHP, Java (multi-line end)
                '"""',      # Python (multi-line)
                "'''",      # Python (multi-line)
                "=begin",   # Ruby (multi-line start)
                # "=end"      # Ruby (multi-line end)
            ]
            # use comment starters to support all languages
            for i in range(len(code_tokens)):
                is_comment = False
                for starter in comment_starters:
                    if code_tokens[i].startswith(starter):
                        is_comment = True
                if not is_comment:
                    yield code_tokens[i]

        for i in range(len(examples["code_tokens"])):
            if "source" in examples:
                # examples["source"][i] = " ".join(
                #     [
                #         t
                #         for t in examples["code_tokens"][i]
                #         if not (t.startswith("//") or t.startswith("/*"))
                #     ]
                # )
                examples["source"][i] = " ".join(remove_comments(examples["code_tokens"][i]))
                examples["source"][i] = all_ascii(examples["source"][i])

            examples["target"][i] = " ".join(examples["docstring_tokens"][i])
        return examples

    def code_from_sample(self, sample):
        return sample["source"]

    def postprocess(self, subset):
        self.subsets[subset] = self.subsets[subset].add_column(
            f"source", [""] * len(self.subsets[subset])
        )

        self.subsets[subset] = self.subsets[subset].add_column(
            f"target", [""] * len(self.subsets[subset])
        )

        # alternative, setting transform function
        # https://huggingface.co/docs/datasets/use_dataset
        self.subsets[subset] = self.subsets[subset].map(
            self.align_sample,
            batched=True,
        )

    def seq2seq_task(self, batch, g_batch, graphs):
        # batch = self.on_before_tokenize(batch)

        base_model_input = BatchEncoding()
        # A way to only use text from graphs
        if self.add_graphs and self.args.cache_enforce:
            # This is for getting the textual representation and processing through
            # tokenizer which would potentially break the sequence again
            # sources = [self.tokenizer.decode(g.x[g.code_mask]) for g in graphs]
            # base_model_input = self.tokenize_source(sources)
            # This is for grammar tokenization, using the graphs to generate encodings
            grammar_tokenization, attention_mask = self.grammar_tokenization(g_batch)
            base_model_input["input_ids"] = grammar_tokenization
            base_model_input["attention_mask"] = attention_mask.to(torch.int64)
        else:
            base_model_input = self.tokenize_source([b["source"] for b in batch])

        base_model_output = self.tokenize_target([b["target"] for b in batch])
        base_model_input["labels"] = base_model_output.input_ids
        base_model_input["decoder_attention_mask"] = base_model_output.attention_mask

        base_model_input = self.on_after_tokenize(base_model_input)

        return (
            # Code text encoded
            base_model_input,
            # Code Graphs
            g_batch,
        )
