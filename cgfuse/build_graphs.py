"""Stage 1: build the code graph cache (AST + data-flow edges) for a dataset.

Writes `<out>/<dataset>_<language>/{train,validation,test}_graphs.pkl` and
`syntax_nodes.pkl`, which the training tasks load (`dataset.cache_enforce=True`).

    python -m cgfuse.build_graphs --dataset concode --language java
    python -m cgfuse.build_graphs --dataset codesearchnet --language java
"""
import argparse
import multiprocessing as mp
import os
import pickle
import time
import traceback
from functools import partial
from pathlib import Path

from datasets import load_dataset
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .utils import Seeder

CONFIG_DIR = str(Path(__file__).parent / "config" / "task" / "dataset")

_dataset = None


def get_config(ds_name, pl_language):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name=ds_name,
            overrides=[f"pl_language={pl_language}", f"preprocessor={pl_language}"],
        )
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.language_task = "seq2seq"
    cfg.debug = False
    cfg.add_graphs = True
    cfg.model_name = "codet5"  # graphs are tokenizer independent; only needed to instantiate
    cfg.num_workers = 1
    cfg.add_edge_mask = False
    return cfg


def init_worker(ds_name, pl_language):
    global _dataset
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cfg = get_config(ds_name, pl_language)
    _dataset = instantiate(cfg.type, cfg, None, Seeder(42))


def build_shard(shard):
    """Returns ({sample_id: pickled networkx graph | None}, {sample_id: error}, syntax node names)."""
    cache, failed, syntax_nodes = {}, {}, set()
    for _, sample in shard.iterrows():
        sample_id = int(sample["_cgf_id"])
        code = sample["code"]
        if _dataset.name == "codesearchnet" and _dataset.pl_language == "python":
            code = code.replace("\t", "  ")
        elif _dataset.name == "concode":
            # CONCODE masks string literals as "srini_string"
            code = code.replace("srini_string", "str").replace('"str" = "str"', '"str"')

        graph_bin = None
        try:
            gt = _dataset.preprocessor.get_graph(code, _dataset.preprocessor.parser)
            for _, attr in gt.nodes(data=True):
                if attr["flat_name"].startswith("AST#"):
                    syntax_nodes.add(attr["flat_name"])
                if attr["flat_name"].startswith("AST#ERROR"):
                    failed[sample_id] = "AST#ERROR"
                for key in ("start_point", "end_point", "ast_type", "order"):
                    del attr[key]
            for _, _, attr in gt.edges(data=True):
                del attr["color"], attr["width"]
            graph_bin = pickle.dumps(gt)
        except BaseException:
            failed[sample_id] = traceback.format_exc()
        cache[sample_id] = graph_bin
    return cache, failed, syntax_nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="concode", choices=["concode", "codesearchnet"])
    parser.add_argument("--language", default="java")
    parser.add_argument("--out", default=".dataset")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--subsets", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N samples per subset (testing)")
    args = parser.parse_args()

    cfg = get_config(args.dataset, args.language)
    if args.dataset == "concode":
        subsets = load_dataset(cfg.source, verification_mode="no_checks")
    else:
        subsets = load_dataset(cfg.source, args.language, verification_mode="no_checks")

    out = Path(args.out) / f"{args.dataset}_{args.language}"
    out.mkdir(parents=True, exist_ok=True)
    all_syntax_nodes = set()
    with mp.get_context("fork").Pool(
        processes=args.workers, initializer=partial(init_worker, args.dataset, args.language)
    ) as pool:
        for subset in args.subsets:
            start = time.time()
            ds = subsets[subset]
            if args.limit:
                ds = ds.select(range(min(args.limit, len(ds))))
            # Same ids as the training datasets use: the dataset's own "id" if present, else the row index
            ids = ds["id"] if "id" in ds.column_names else list(range(len(ds)))
            ds = ds.add_column("_cgf_id", [int(i) for i in ids])
            n_shards = max(1, min(len(ds), args.workers * 3))
            shards = [ds.shard(num_shards=n_shards, index=i).to_pandas() for i in range(n_shards)]

            cache, failed = {}, {}
            for res, fail, nodes in pool.imap_unordered(build_shard, shards):
                cache.update(res)
                failed.update(fail)
                all_syntax_nodes.update(nodes)
            cache = dict(sorted(cache.items()))

            with open(out / f"{subset}_graphs.pkl", "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"{subset}: {len(cache)} graphs, {len(failed)} failed/with AST#ERROR "
                f"{sorted(failed)[:20]}{' ...' if len(failed) > 20 else ''} "
                f"({time.time() - start:.0f}s)"
            )

    with open(out / "syntax_nodes.pkl", "wb") as f:
        pickle.dump(all_syntax_nodes, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"syntax nodes: {len(all_syntax_nodes)} -> {out}")


if __name__ == "__main__":
    main()
