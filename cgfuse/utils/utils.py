import torch
import random
import os
import numpy as np
import pwd
import hashlib
import dill
import pickle
from itertools import islice
from transformers import set_seed
from prettytable import PrettyTable


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seeder:
    def __init__(self, seed=1993) -> None:
        """Sets seed for reproducability. This object also
        generates Generator objects with base seed if requested

        Args:
            seed (int, optional): Defaults to 1993.
        """
        self.seed = seed
        set_seed(self.seed)
        random.seed(seed)
        os.environ["PYHTONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True

    def torch_gen(self, device="cpu"):
        gen = torch.Generator(device=device)
        return gen.manual_seed(self.seed)

    def np_gen(self):
        return np.random.default_rng(self.seed)


def get_current_user() -> str:
    return pwd.getpwuid(os.getuid()).pw_name


def hash(string: str, hash_method: str = "short") -> str:
    """
    Takes a string with arbitrary length and returns the hash of it encoded in utf-8
    """
    if hash_method == "long":
        return hashlib.sha224(bytes(string, encoding="utf-8")).hexdigest()
    else:
        return hashlib.md5(bytes(string, encoding="utf-8")).hexdigest()


def compare_models(module_1, module_2):
    """Compares two modules to see if they are same
    got from: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/6

    Args:
        model_1 (torch.nn.Module): lhs module
        model_2 (torch.nn.Module): rhs module

    Raises:
        Exception: Models have different layers altogether

    Returns:
        int: Return count of discrepancies between modules
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        module_1.state_dict().items(), module_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] != key_item_2[0]:
                raise Exception
            # else:
            #     print("Mismatch found at ", key_item_1[0])

    return models_differ


def compare_models2(module_1, module_2):
    """This is not finished but used to understand modules.ED_from_encoder_model
    initialization and its difference from transformers.EncoderDecoderModel.from_encoder_decoder_pretrained.
    """
    models_differ = 0
    m2_dict = module_2.state_dict()
    for key_item_1 in module_1.state_dict().items():
        m2_key = f"roberta.{key_item_1[0]}"
        if m2_key in m2_dict:
            if torch.equal(key_item_1[1], m2_dict[m2_key]):
                pass
            else:
                models_differ += 1
                print(f"Mismatch found at {key_item_1[0]} & {m2_key}")
        elif key_item_1[0] in m2_dict:
            if torch.equal(key_item_1[1], m2_dict[key_item_1[0]]):
                pass
            else:
                models_differ += 1
                print(f"Mismatch found at {key_item_1[0]} & {key_item_1[0]}")
        else:
            print(f"{key_item_1[0]} not in module_2")

    return models_differ


def count_parameters(model) -> int:
    """Check trainable parameters in model
    https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    Args:
        model (_type_): _description_

    Returns:
        int: total trainable parameter count
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    table.add_row(["Total Trainable Params", total_params])
    print(table)
    return total_params


def add_to_dict(_dict, elem):
    if elem not in _dict:
        _dict[elem] = len(_dict)


def get_size_count(input_dict, chunks):
    n = len(input_dict)
    size = n // chunks + (n % chunks > 0)
    chunk_count = -(-n // size)
    return size, chunk_count


def split_dict_equally(input_dict, chunks):
    from itertools import islice

    """Split dictionary into roughly equal-sized chunks without loading all items into memory at once."""
    iter_items = iter(input_dict.items())  # Lazy iterator (no list conversion)
    n = len(input_dict)
    size = -(-n // chunks)

    for _ in range(chunks):
        # Slice the iterator on-the-fly
        chunk = dict(islice(iter_items, size))
        if chunk:  # Skip empty chunks
            yield chunk


def split_dict_batches(input_dict, size):
    """Split dictionary into roughly equal-sized chunks without loading all items into memory at once."""
    iter_items = iter(input_dict.items())  # Lazy iterator (no list conversion)
    n = len(input_dict)
    chunks = -(-n // size)

    for _ in range(chunks):
        # Slice the iterator on-the-fly
        chunk = dict(islice(iter_items, size))
        if chunk:  # Skip empty chunks
            yield chunk


def debug_mmap(s_print=False):
    # Check mmap limit
    vm_max_map_count = -1
    with open("/proc/sys/vm/max_map_count") as f:
        vm_max_map_count = int(f.read().strip())
        if s_print:
            print("vm.max_map_count:", vm_max_map_count)

    # Check current usage
    mmap_count = -1
    with open(f"/proc/{os.getpid()}/maps") as f:
        mmap_count = len(f.readlines())
        if s_print:
            print("Current memory-mapped regions:", mmap_count)

    return vm_max_map_count, mmap_count


_worker_func = None


def init_worker(serialized_func):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _worker_func
    _worker_func = dill.loads(serialized_func)


def wrapper(arg):
    return _worker_func(arg)


def get_syntax_nodes(files_prefix):
    if not os.path.exists(f"{files_prefix}/syntax_nodes.pkl"):
        return None

    with open(f"{files_prefix}/syntax_nodes.pkl", "rb") as f:
        syntax_nodes = pickle.load(f)

    return syntax_nodes

def mp_load_dataset(
    files_prefix,
    tokenizer,
    process_count=32,
    add_edge_mask=True,
    subsets=["train", "validation", "test"],
    chunk_size=5000,
    batch_size=100,
    debug=False,
):
    """
    1. Read the code graphs
    2. Tokenize (with target model's tokenizer) code tokens and create connections
      2.1. Create super nodes and make all connections if some breaks down.
    3. Create pytorch_geometric.Data instances for each sample
    """
    from .preprocessors.preprocessor import Preprocessor
    from functools import partial
    import multiprocessing as mp
    import gc, copy
    from tqdm import tqdm


    get_pyg_data_fromgraph = partial(
        Preprocessor.get_pyg_data_fromgraph,
        add_edge_mask=add_edge_mask,
    )
    graph_tokenize = partial(Preprocessor.graph_tokenize, tokenizer=tokenizer)

    cache = {}
    if debug:
        for subset in subsets:
            cache[subset] = {}
            with open(f"{files_prefix}/{subset}_graphs.pkl", "rb") as f:
                cached = pickle.load(f)
                # If debug, get representative 128 samples to work on
                # Same implemented in Dataset.get_dataloader as well.
                for i in range(128):
                    if not cached[i]:
                        cache[subset][i] = None
                        continue
                    graph = pickle.loads(cached[i])
                    new_G = graph_tokenize(graph)
                    py_graph = get_pyg_data_fromgraph(new_G)
                    cache[subset][i] = py_graph
        return cache

    def worker_task(_cached):
        in_cache = {}
        for idx, graph in _cached.items():
            if not graph:
                in_cache[idx] = None
                continue
            graph = pickle.loads(graph)

            # Get code graph adapted to tokenizer
            new_G = graph_tokenize(graph)
            py_graph = get_pyg_data_fromgraph(new_G)

            in_cache[idx] = py_graph
        return in_cache

    serialized_func = dill.dumps(worker_task)

    with mp.get_context("fork").Pool(
        processes=process_count,
        initializer=init_worker,
        initargs=(serialized_func,),
    ) as pool:
        for subset in subsets:
            cache[subset] = {}
            with open(f"{files_prefix}/{subset}_graphs.pkl", "rb") as f:
                cached = pickle.load(f)

            # Divide into chunks first to make sure mmap is not filled
            i = 0
            for chunk in tqdm(
                split_dict_batches(cached, chunk_size),
                desc=f"{subset}",
                total=-(-len(cached) // chunk_size),
            ):
                for result in tqdm(
                    pool.imap_unordered(wrapper, split_dict_batches(chunk, batch_size)),
                    total=-(-len(chunk) // batch_size),
                    position=1,
                    desc=f"Chunk {i}",
                    leave=False,
                ):
                    # Copy all intances returned from workers since they fill mmap and cause crash otherwise.
                    # Likely bug due to tensor and torch_geometric.Data creation through workers.
                    cache[subset].update({idx: copy.deepcopy(el) for idx, el in result.items()})
                    del result

                # Make sure that mmap and memory is cleaned
                gc.collect()
                i += 1

    return cache
