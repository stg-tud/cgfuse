# CGFuse: Deep Graph-Language Fusion for Structure-Aware Code Generation

Code for the FORGE '26 paper
**[Deep Graph-Language Fusion for Structure-Aware Code Generation](https://doi.org/10.1145/3793655.3793724)**
by Mert Tiftikci, Amir Molzam Sharifloo and Mira Mezini ([arXiv:2605.03689](https://arxiv.org/abs/2605.03689)).

CGFuse pre-trains a graph neural network (GNN) on code graphs (ASTs augmented with
data-flow edges) and fuses its node representations into intermediate layers of a
pre-trained language model at the token level:

$$h^l_f = h^l_\theta + \lambda \, h^l_g$$

where $h^l_\theta$ is the transformer's output at layer $l$, $h^l_g$ the GNN's output for
the nodes in the current context, and $\lambda$ the fusion strength.

## Setup

```bash
conda create -n cgfuse python=3.12 && conda activate cgfuse
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA
pip install -r requirements.txt
```

All commands below run from the repository root. Outputs go to `.dataset/` (graph cache),
`.logs/` (checkpoints) and `.aim/` (metrics; browse with `aim up`).
Add `debug=True` to any training command for a quick run on 128 samples without logging
(and `trainer.accelerator=cpu` on a machine without a GPU).

## The three stages

### 1. Build the code graphs

```bash
python -m cgfuse.build_graphs --dataset concode --language java          # fusion / evaluation data
python -m cgfuse.build_graphs --dataset codesearchnet --language java    # GNN pre-training data
```

Each sample's code is parsed with tree-sitter into an AST and augmented with
`comingFrom` / `calculatedBy` data-flow edges (`cgfuse/utils/preprocessors/`). The graphs are
tokenizer independent; sub-tokens are created at training time with the target model's tokenizer.
A handful of samples cannot be parsed and are excluded (see `graph_not_generated` in
`cgfuse/utils/datasets/`).

### 2. Pre-train the GNN expert

The GNN is pre-trained on node classification for each target PLM (its embeddings are
initialised from that PLM's token embeddings, so an expert belongs to one PLM):

```bash
python -m cgfuse task=bs_gnn_jv \
    task/model@model=PLBART \
    task/gnn@gnn=rgcn \
    task/dataset@dataset=codesearchnet dataset.pl_language=java \
    gnn.intermediate=[] gnn.activation=gelu
```

- `task/gnn@gnn`: `rgcn`, `graphsage` or `gin`.
- `gnn.intermediate`: `[]` is a single GNN layer (used for the main results);
  `[512]` and `[512,512]` give the 2- and 3-layer variants of Table 4.
- `gnn.activation`: `gelu` for the BERT/BART families, `relu` for T5/CodeT5.

The checkpoint is written to `.logs/bootstrap/gnn_java/ckpt/<run hash>/last.ckpt`.

### 3. Fuse the GNN into the PLM and fine-tune

```bash
python -m cgfuse task=cgfuse \
    task/model@model=PLBART \
    task/trainer/optimizer@trainer.optimizer=PLBART \
    task/trainer/scheduler@trainer.scheduler=PLBART \
    trainer.fusion.expert=.logs/bootstrap/gnn_java/ckpt/<run hash>/last.ckpt \
    trainer.fusion.target_layers=[11] \
    trainer.fusion.scale=1.0
```

Training first adapts only the attached GNN (`trainer.fusion.adapting_epoch`), then fine-tunes
the PLM and GNN jointly (`trainer.fusion.epoch`), with early stopping on validation CodeBLEU.
The best checkpoint is evaluated on the CONCODE test set (EM, BLEU, CodeBLEU).

`trainer.fusion.scale` is $\lambda$. `trainer.fusion.target_layers` indexes the decoder's layers
for the encoder models (which are turned into encoder-decoder models), and the
encoder+decoder layer stack for the BART and T5 families. Settings of the main experiments:

| PLM | `task/model@model` | optimizer / scheduler | GNN (Table 2) | λ | `target_layers` |
|---|---|---|---|---|---|
| BERT | `bert` | `bert` | R-GCN | 1 | `[0,1,2]` |
| RoBERTa | `roberta` | `roberta` | GIN | 0.5 | `[11]` |
| BART | `bart` | `bart` | GIN | 0.5 | `[6,7]` |
| T5 | `t5` | `t5` | GIN | 0.5 | `[23]` |
| CodeBERT | `CodeBERT` | `CodeBERT` | GIN | 1 | `[0,1,2,3]` |
| GraphCodeBERT | `GraphCodeBERT` | `GraphCodeBERT` | GraphSAGE | 1 | `[0,11]` |
| UniXcoder | `unixcoder` | `unixcoder` | GIN | 1 | `[0,11]` |
| PLBART | `PLBART` | `PLBART` | R-GCN | 1 | `[11]` |
| CodeT5 | `codeT5_M220` | `codeT5` | GIN | 1 | `[12,13,22,23]` |

### Baselines

The same PLMs fine-tuned without a GNN:

```bash
python -m cgfuse task=baseline task/model@model=PLBART \
    task/trainer/optimizer@trainer.optimizer=PLBART task/trainer/scheduler@trainer.scheduler=PLBART
```

## Layout

```
cgfuse/
  __main__.py          Hydra entry point (python -m cgfuse task=...)
  build_graphs.py      stage 1: code graph cache
  config/              Hydra configs: task/{baseline,bs_gnn_jv,cgfuse}.yaml + model/gnn/dataset/trainer groups
  tasks/               bootstrap.py (GNN pre-training), struct_infuse.py (fusion), baseline.py
  models/              GNN experts (R-GCN, GraphSAGE, GIN) and encoder -> encoder-decoder conversion
  utils/adapter.py     the fusion adapter that injects GNN features into transformer layers
  utils/preprocessors/ tree-sitter AST + data-flow graph extraction per language
  utils/datasets/      CONCODE and CodeSearchNet loaders
  utils/metrics/       EM, BLEU, CodeBLEU
```

## Notes

- Rebuilding the graphs with the pinned tree-sitter versions reproduces the caches used for the paper
  (checked for CONCODE and CodeSearchNet Java), except for a handful of samples (4 of the 100k CONCODE
  training samples) whose non-ASCII character literals are now tokenized as `<unk>` instead of `?`.
- All results in the paper come from a single run with seed 42.
- `cgfuse/utils/metrics/` vendors third-party evaluation code under its original licenses:
  CodeBLEU (Microsoft / [k4black/codebleu](https://github.com/k4black/codebleu), MIT), NLTK's BLEU
  (Apache 2.0), Google's BLEU (Apache 2.0) and the smoothed BLEU of CodeXGLUE.

## Citation

```bibtex
@inproceedings{tiftikci2026cgfuse,
  title     = {Deep Graph-Language Fusion for Structure-Aware Code Generation},
  author    = {Tiftikci, Mert and Molzam Sharifloo, Amir and Mezini, Mira},
  booktitle = {Proceedings of the 2026 IEEE/ACM Third International Conference on AI Foundation Models and Software Engineering (FORGE '26)},
  year      = {2026},
  address   = {Rio de Janeiro, Brazil},
  publisher = {ACM},
  doi       = {10.1145/3793655.3793724}
}
```
