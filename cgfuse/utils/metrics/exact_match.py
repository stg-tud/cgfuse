from . import Metric

# Implementation from NIU, Changan, et al.
# An empirical comparison of pre-trained models of source code. arXiv preprint arXiv:2302.04026, 2023.
# https://github.com/NougatCA/FineTuner


class ExactMatch(Metric):
    def __init__(self, prefix='', tokenizer=None) -> None:
        super().__init__()
        self.prefix = prefix
        self.tokenizer = tokenizer

    def __call__(self, preds, golds) -> dict:
        assert len(preds) == len(golds)
        count = 0
        for pred, gold in zip(preds, golds):
            if pred == gold:
                count += 1
        avg_score = count / len(preds)
        return avg_score * 100
