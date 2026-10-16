#!/bin/bash
# Main results (Table 2): for each PLM, pre-train its GNN expert on CodeSearchNet Java graphs,
# then fuse it into the PLM's last layer and fine-tune on CONCODE.
#   bash scripts/main_results.sh                 # all nine PLMs
#   bash scripts/main_results.sh PLBART bert     # a subset
# Needs the graph caches: python -m cgfuse.build_graphs --dataset {concode,codesearchnet} --language java
set -euo pipefail

#        model          optimizer/scheduler  GNN        lambda  activation  last layer
SETTINGS="
bert            bert            rgcn       1    gelu  11
roberta         roberta         gin        0.5  gelu  11
bart            bart            gin        0.5  gelu  11
t5              t5              gin        0.5  relu  23
CodeBERT        CodeBERT        gin        1    gelu  11
GraphCodeBERT   GraphCodeBERT   graphsage  1    gelu  11
unixcoder       unixcoder       gin        1    gelu  11
PLBART          PLBART          rgcn       1    gelu  11
codeT5_M220     codeT5          gin        1    relu  23
"
DEVICE=${DEVICE:-0}

while read -r model opt gnn scale act layer <&3; do
    [ -z "$model" ] && continue
    if [ $# -gt 0 ] && [[ ! " $* " =~ " $model " ]]; then continue; fi

    expert_dir=.logs/experts/${model}_${gnn}
    python -m cgfuse task=bs_gnn_jv workdir=$expert_dir \
        task/model@model=$model task/gnn@gnn=$gnn \
        task/dataset@dataset=codesearchnet dataset.pl_language=java \
        "gnn.intermediate=[]" gnn.activation=$act "trainer.device=[$DEVICE]"
    expert=$(ls -t $(find $expert_dir -name last.ckpt) | head -1)

    python -m cgfuse task=cgfuse workdir=.logs/fusion/${model}_${gnn} \
        task/model@model=$model \
        task/trainer/optimizer@trainer.optimizer=$opt task/trainer/scheduler@trainer.scheduler=$opt \
        trainer.fusion.expert=$expert "trainer.fusion.target_layers=[$layer]" trainer.fusion.scale=$scale \
        "trainer.device=[$DEVICE]"
done 3<<< "$SETTINGS"
