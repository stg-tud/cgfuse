from datasets import Dataset

start_offsets = {
    "t5": { "input": 0, "output": 1 },
}

def batch_adapt(dataset: Dataset, model_name: str):
    """ Set on_before_tokenize and on_after_tokenize for dataset according to
    special model preparation
    """
    setattr(dataset.__class__, "on_before_tokenize", globals()[f"{model_name}_before"])
    setattr(dataset.__class__, "on_after_tokenize", globals()[f"{model_name}_after"])
    # default offset is 1 but there might be special cases
    # codebert, graphcodebert, unixcoder, codet5, codet5p has <s> and any possible additional tokens listed below
    # plbart does not have <s> in the beginning but any other special below.
    return start_offsets[model_name] if model_name in start_offsets else { "input": 0, "output": 1 }


def bart_before(dataset, inp, output):
    return inp, output


def bart_after(dataset, model_input):
    return model_input


def roberta_before(dataset, inp, output):
    return inp, output


def roberta_after(dataset, model_input):
    return model_input


def bert_before(dataset, inp, output):
    return inp, output


def bert_after(dataset, model_input):
    return model_input


def unixcoder_before(dataset, inp, output):
    return inp, output


def unixcoder_after(dataset, model_input):
    return model_input


def codebert_before(dataset: Dataset, inp, output):
    return inp, output


def codebert_after(dataset: Dataset, model_input):
    return model_input


def graphcodebert_before(dataset: Dataset, inp, output):
    return inp, output


def graphcodebert_after(dataset: Dataset, model_input):
    return model_input


def t5_before(dataset, inp, output):
    return inp, output


def t5_after(dataset, model_input):
    return model_input

def codet5_before(dataset: Dataset, inp, output):
    return inp, output


def codet5_after(dataset: Dataset, model_input):
    return model_input


def plbart_before(dataset: Dataset, inp, output):
    return inp, output


def plbart_after(dataset: Dataset, model_input):
    return model_input
