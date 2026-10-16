#!/bin/bash
# Baselines (Table 2): each PLM fine-tuned on CONCODE without a GNN.
#   bash scripts/baselines.sh                    # all nine PLMs
#   bash scripts/baselines.sh PLBART             # a subset
set -euo pipefail

#       model           optimizer/scheduler
SETTINGS="
bert            bert
roberta         roberta
bart            bart
t5              t5
CodeBERT        CodeBERT
GraphCodeBERT   GraphCodeBERT
unixcoder       unixcoder
PLBART          PLBART
codeT5_M220     codeT5
"
DEVICE=${DEVICE:-0}

while read -r model opt <&3; do
    [ -z "$model" ] && continue
    if [ $# -gt 0 ] && [[ ! " $* " =~ " $model " ]]; then continue; fi
    python -m cgfuse task=baseline workdir=.logs/baseline/$model task/model@model=$model \
        task/trainer/optimizer@trainer.optimizer=$opt task/trainer/scheduler@trainer.scheduler=$opt \
        "trainer.device=[$DEVICE]"
done 3<<< "$SETTINGS"
