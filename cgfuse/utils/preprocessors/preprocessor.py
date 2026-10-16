import importlib

import networkx as nx
from networkx import MultiGraph
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from tree_sitter import Language, Parser

from ..utils import add_to_dict
from datasets import Dataset
from .graph_extractor import GraphExtractor


def all_ascii(text: str) -> str:
    """

    Args:
        string (str): code to be non-ascii characters removed

    Returns:
        str: ascii version
    """
    return "".join([t if (ord(t) < 128 and t != "\x00") else "<unk>" for t in text])
    # return "".join([t if (ord(t) < 128 and t != "\x00") else "?" for t in text])


class Preprocessor:
    KEEP_TYPES = [
        "identifier",
        "type_identifier",
        "string",
        "null_literal",
        "decimal_floating_point_literal",
        "character_literal",
        "hex_integer_literal",
        "string_literal",
        "string_fragment", # Added after special treatment in CodeSearchNet Java
        "escape_sequence", # Added after special treatment in CodeSearchNet Java
        "binary_integer_literal", # Added after special treatment in CodeSearchNet Java
        "underscore_pattern", # ? Added after special treatment in CodeSearchNet Java
        "boolean_type",
        "decimal_integer_literal",
        "octal_integer_literal",
        "void_type",
        "void",
    ]

    VIS_OPTIONS = {
        "parent": {"color": "black", "width": 1},
        "comesFrom": {"color": "red", "width": 2},
        "computedFrom": {"color": "blue", "width": 3},
        "type": {"color": "violet", "width": 2},
    }

    TREE_SITTER_BASE_PATH = "https://github.com/tree-sitter/tree-sitter-"

    def __init__(self, args) -> None:
        self.args = args
        # Should be set in prepare
        self.debug = False
        self.dataset_name = None
        self.tokenizer = None
        self.parser = None
        # Set in the language specific implementations
        self.language = None

        self.vocab = {}
        self.ast_vocab = {}
        self.code_types = {}

        self.centroid = self.args.centroid
        self.add_types = self.args.add_types
        self.add_code_mask = self.args.add_code_mask
        # Only meaningful when graphs are decoded.
        self.add_edge_mask = self.args.add_edge_mask

    def prepare(self, dataset: Dataset, tokenizer=None):
        self.dataset_name = dataset.__class__.__name__
        self.debug = dataset.args.debug
        self.tokenizer = tokenizer
        self.language = dataset.pl_language
        self.edge_types = {t: i for i, t in enumerate(Preprocessor.VIS_OPTIONS.keys())}

        self.vocab = self.tokenizer.get_vocab() if self.tokenizer is not None else {}
        self.ast_vocab = self.ast_tokens()

        language_module = self.get_language(self.language.replace('-', '_'))
        self.parser = Parser(language_module)

    def ast_tokens(self):
        if self.tokenizer is not None and "unixcoder" in self.tokenizer.name_or_path:
            return {k.split("#")[1]: v for k, v in self.vocab.items() if "AST#" in k}
        return {}

    def get_language(self, lang_name):
        """Get the language object from a tree-sitter language module."""
        try:
            language_module = importlib.import_module(f"tree_sitter_{lang_name}")

            # Try different ways to access the language
            if hasattr(language_module, 'language'):
                if callable(language_module.language):
                    return Language(language_module.language())
                else:
                    return Language(language_module.language)
            elif hasattr(language_module, 'Language'):
                return Language(language_module.Language())
            else:
                # For PHP and other modules with specific language functions
                if lang_name == 'php' and hasattr(language_module, 'language_php'):
                    return Language(language_module.language_php())

                # For some modules, the language might be directly available
                attrs = [attr for attr in dir(language_module) if not attr.startswith('_')]
                print(f"Available attributes in tree_sitter_{lang_name}: {attrs}")
                return None
        except Exception as e:
            print(f"Error loading language {lang_name}: {e}")
            return None

    def get_pyg_data(self, code, vocab=None) -> Data:
        # Get the code graph
        graph = self.get_graph(code, self.parser)
        # get pytorch graph, maybe added edge_mask and stripped
        py_graph = Preprocessor.get_pytorch_graph(graph, add_edge_mask=self.add_edge_mask)

        # Try to encode nodes
        if self.tokenizer is not None:
            # Return encoded, according to given tokenizer
            x = torch.tensor(self.tokenizer.convert_tokens_to_ids(py_graph.flat_name))
        elif vocab is not None:
            # If vocab is provided, assumed to be curated over generating
            x = torch.tensor([vocab[fname] for fname in py_graph.flat_name])
        else:
            # Get direct name instead of encoded
            x = py_graph.flat_name

        # Create base dict for data
        data = Data(
            x=x,
            y=x,
            edge_type=py_graph.y,
            edge_index=py_graph.edge_index,
            code_mask=py_graph.code_token,
            code_len=torch.sum(py_graph.code_token),
        )
        if self.add_edge_mask:
            data["edge_mask"] = py_graph.edge_mask
        return data

    @staticmethod
    def get_pyg_data_fromgraph(graph, vocab=None, tokenizer=None, add_edge_mask=True) -> Data:
        # get pytorch graph, maybe added edge_mask and stripped
        py_graph = Preprocessor.get_pytorch_graph(graph, add_edge_mask=add_edge_mask)

        # Try to encode nodes
        if "t_id" in py_graph:
            x = py_graph.t_id
        elif tokenizer is not None:
            # Return encoded, according to given tokenizer
            x = torch.tensor(tokenizer.convert_tokens_to_ids(py_graph.flat_name))
        elif vocab is not None:
            # If vocab is provided, assumed to be curated over generating
            x = torch.tensor([vocab[fname] for fname in py_graph.flat_name])
        else:
            # Get direct name instead of encoded
            x = py_graph.flat_name

        # Create base dict for data
        data = Data(
            x=x,
            y=x,
            edge_type=py_graph.y,
            edge_index=py_graph.edge_index,
            code_mask=py_graph.code_token,
            code_len=torch.sum(py_graph.code_token),
        )
        if add_edge_mask:
            data["edge_mask"] = py_graph.edge_mask

        del py_graph
        return data

    def get_graph(self, code, parser) -> MultiGraph:
        """
        Get the graph representation of a code snippet

        :param code: code snippet
        :param parser: tree-sitter parser

        :return: graph representation (with dataflow) of the code snippet
        """
        (
            root_node,  # tree-sitter AST root node
            code_tokens,
            index_to_code,  # Text2Code mapping
            DFG,  # Dataflow connections
            final_code,
        ) = GraphExtractor.extract_graph(
            all_ascii(code),
            parser,
            self.extract_dfg,
            self.language,
            self.dataset_name,
        )

        # convert tree-sitter AST to networkx graph, represented by root_node
        G, flattened, node2token = self.ts2nx(
            root_node, index_to_code, self.language == "java"
        )

        # add dataflow relations as edges to graph
        G = self.add_DFG_relations(
            G,
            filter(lambda x: len(x[4]) > 0, DFG),
            index_to_code,
            self.language == "java",
        )

        return G

    @staticmethod
    def get_pytorch_graph(G, add_edge_mask=True, strip=True) -> Data:
        """
        Get the pytorch geometric graph representation of a networkx graph

        :param G: networkx graph
        :param add_edge_mask: masks for the edges in the generation, similar to attention mask
        :param strip: remove unnecessary fields

        :return: pytorch geometric graph representation that reduces the size of the graph and removes unnecessary attributes
        """
        py_graph = from_networkx(G)

        if add_edge_mask:
            if "order" not in py_graph:
                py_graph.order = torch.arange(py_graph.code_token.shape[0])
            code_token_order = py_graph.order[py_graph.code_token]
            edge_mask = torch.full_like(py_graph.y, -1)
            last_i = -1
            j = 0
            for i, cursor in enumerate(code_token_order):
                j_temp = j
                while True:
                    if j_temp == len(py_graph.edge_index[1]):
                        last_i = i
                        break
                    k = py_graph.edge_index[1][j_temp]
                    j_temp += 1
                    if cursor == k:
                        j = j_temp
                        break
                mask = py_graph.edge_index[1][:j] <= cursor
                residual_mask = edge_mask[:j] == -1
                edge_mask[:j][mask & residual_mask] = i
                # py_graph.edge_index = py_graph.edge_index[:,:j][:,mask]

            if last_i != -1:
                edge_mask[edge_mask == -1] = last_i

            py_graph["edge_mask"] = edge_mask

        # Remove unnecessary
        if strip:
            del (
                py_graph.color,
                py_graph.width,
                py_graph.order,
                py_graph.start_point,
                py_graph.end_point,
                py_graph.first_token,
            )

        return py_graph

    def extract_dfg(self):
        raise NotImplementedError("The method is not implemented for the language ...")

    def add_DFG_relations(self, G, relations, index_to_code, skip_first=True):
        """
        Add data flow relations to the graph

        :param G: graph to add relations to
        :param relations: relations to add
        :param index_to_code: mapping from index to code
        :param skip_first: whether to skip the first line

        :return: graph with added relations

        """
        # Add dfg connections
        positions = list(index_to_code.keys())

        pos2node = {}
        # create a mapping from positions to node names at that position
        for name, attr in G.nodes(data=True):
            if (
                self.add_types
                and attr["end_point"][0] == -1
                and attr["end_point"][1] == -1
            ):
                continue
            java_skip = (1 if skip_first else 0)
            k = (
                (
                    attr["start_point"][0] - java_skip,
                    attr["start_point"][1],
                ),
                (attr["end_point"][0] - java_skip, attr["end_point"][1]),
            )
            if k not in index_to_code:
                # maybe added space before
                k = (
                    (
                        attr["start_point"][0] - java_skip,
                        attr["start_point"][1] - 1,
                    ),
                    (attr["end_point"][0] - java_skip, attr["end_point"][1]),
                )
            if k in pos2node:
                pos2node[k].append(name)
            else:
                pos2node[k] = [name]

        for token, loc, _type, t_tokens, t_locs in relations:
            if len(t_tokens) == 0:
                continue

            node_names = pos2node[positions[loc]]
            for t_loc in t_locs:
                t_node_names = pos2node[positions[t_loc]]
                for t_node_name in t_node_names:
                    for node_name in node_names:
                        G.add_edge(
                            node_name,
                            t_node_name,
                            y=self.edge_types[_type],
                            **Preprocessor.VIS_OPTIONS[_type],
                        )

        return G

    def flatten(self, tokens, _format, strip_graph=True):
        """
        Flatten the tokens to a string, optionally stripping tokens from vocab and replacing graph tokens with spaces

        :param tokens: tokens to flatten
        :param _format: format of the tokens
        :param strip_graph: whether to strip the graph tokens

        :return: flattened string
        """

        if strip_graph:
            return (
                "".join(
                    [
                        t
                        for t in tokens
                        if t
                        not in (
                            self.vocab if _format == "unixcoder" else self.ast_vocab
                        )
                    ]
                )
                .replace("Ġ", " ")
                .strip()
            )
        else:
            return "".join([t for t in tokens]).replace("Ġ", " ").strip()

    def ts2nx(self, root, index_to_code, skip_first=True):
        """
        Convert a tree-sitter AST to a networkx graph by creating a MultiGraph and adding nodes and edges

        :param root: root of the AST
        :param index_to_code: mapping from index to code
        :param skip_first: whether to skip the first line

        :return: networkx graph, flattened representation of the AST and mapping from node to token

        """
        G = nx.MultiGraph(first_token=False)
        # Add cls as first node if it is set up
        if self.centroid is not None:
            G.add_node(
                self.centroid,
                flat_name=self.centroid,
                order=-1,
                code_token=False,
                ast_type=self.centroid,
                start_point=(0, 0),
                end_point=(0, 0),
            )

        flattened = []
        self._ts2nx(root, index_to_code, G, flattened, skip_first)

        node2token = [d["order"] for n, d in G.nodes(data=True) if d["order"] != -1]
        node2token = [
            p[0]
            for p in sorted(
                [(i, o) for i, o in enumerate(node2token)], key=lambda p: p[1]
            )
        ]

        # connect all the nodes to center node
        if self.centroid is not None:
            for n in G.nodes:
                G.add_edge(
                    n,
                    self.centroid,
                    y=self.edge_types["parent"],
                    **Preprocessor.VIS_OPTIONS["parent"],
                )

        return G, flattened, node2token

    @staticmethod
    def get_unique_name(name, G):
        """
        Get a unique name for a node

        :param name: name to make unique
        :param G: graph to check for uniqueness

        :return: unique name
        """
        if name not in G:
            return name
        c = 0
        while True:
            name = f"{name}##{c:02d}"
            if name not in G:
                break
            name = name[:-4]
            c += 1
        return name

    @staticmethod
    def gen_name_from(node, index_to_code, add_pos=False, skip_first=True):
        """
        Generate a name for a node and add the position if add_pos is True

        :param node: node to generate name for
        :param index_to_code: mapping from index to code
        :param add_pos: whether to add the position to the name
        :param skip_first: whether to skip the first line

        :return: name of the node and optionally the start point
        """
        name, pos = Preprocessor.gen_name(
            node.start_point, node.end_point, node.type, index_to_code, skip_first
        )
        return f"{name}{pos if add_pos else ''}"

    @staticmethod
    def gen_name(start_point, end_point, _type, index_to_code, skip_first=True):
        """
        Generate a name for a node

        :param start_point: start point of the node (line, column)
        :param end_point: end point of the node (line, column)#
        :param _type: type of the node
        :param index_to_code: mapping from index to code
        :param skip_first: whether to skip the first line

        :return: name of the node and the start point

        """
        # keep name if it is a keep_type, otherwise use the type
        java_skip = (1 if skip_first else 0)
        region = (
            (start_point[0] - java_skip, start_point[1]),
            (end_point[0] - java_skip, end_point[1]),
        )
        if region not in index_to_code:
            # maybe added space before
            region = (
                (start_point[0] - java_skip, start_point[1] -1),
                (end_point[0] - java_skip, end_point[1]),
            )
        name = (
            index_to_code[region][1]
        )
        return name, (start_point[0] - java_skip, start_point[1])

    @staticmethod
    def graph_tokenize(G, tokenizer, order_req=False):
        # ? It might be better to iterate over all nodes to create a new graph
        # ? where code nodes are tokenized, this would preserve ordering most likely
        new_G = nx.MultiGraph(first_token=False)
        super_nodes = {}
        order_shift = 0
        for super_node in list(G.nodes()):
            # skip syntax nodes
            attr = G.nodes[super_node]
            node_tokens = tokenizer.tokenize(attr.get("flat_name"))

            if not attr.get("code_token"):
                # tokenize applied lower case for some reason,
                # we know that the internal values are normalized already!
                t_ids = tokenizer.convert_tokens_to_ids(attr.get("flat_name"))
                if order_req:
                    attr["order"] = attr["order"] + order_shift
                attr["t_id"] = t_ids[0] if isinstance(t_ids, list) else t_ids
                new_G.add_node(super_node, **attr)
                # add edges too
                continue

            t_ids = tokenizer.convert_tokens_to_ids(node_tokens)

            # Tokenize the code token
            super_nodes[super_node] = []

            # Iterate over all nodes
            for i, (node_token, t_id) in enumerate(zip(node_tokens, t_ids)):
                # Create a copy of node attributes to add
                new_attr = attr.copy()
                # Update flat_name since this is a subtext
                new_attr["flat_name"] = node_token
                new_attr["t_id"] = t_id
                if order_req:
                    new_attr["order"] = new_attr["order"] + order_shift
                order_shift += 1

                # Get unique name so Different nodes are not overwritten
                unique_name = Preprocessor.get_unique_name(node_token, new_G)
                # Add the new node with attributes
                new_G.add_node(unique_name, **new_attr)
                super_nodes[super_node].append(unique_name)
            order_shift -= 1

        for u, v, edge_attr in list(G.edges(data=True)):
            u = super_nodes[u] if u in super_nodes else [u]
            v = super_nodes[v] if v in super_nodes else [v]
            for u_alt in u:
                for v_alt in v:
                    new_G.add_edge(u_alt, v_alt, **edge_attr)

        return new_G


    def ignored_nodes(self):
        ignored_nodes = ["escape_interpolation", "\\", "comment"]
        if self.language == "python":
            ignored_nodes.append("escape_sequence")
        if self.language == "php":
            ignored_nodes.pop(1)  # Keep backslash for now
        return ignored_nodes


    def filter_instr_nodes(self, root):
        # These two exists within a string_content node,
        # which breaks the extraction of string literals
        return [c for c in root.children if c.type not in self.ignored_nodes()]


    def _ts2nx(
        self,
        root,
        index_to_code,
        G,
        flattened=None,
        skip_first=True,
        unixcoder=False,
    ):
        """
        Convert a tree-sitter AST to a networkx graph

        :param root: root of the AST
        :param index_to_code: mapping from index to code
        :param G: graph to add nodes to
        :param flattened: flattened representation of the AST
        :param skip_first: whether to skip the first line
        :param unixcoder: whether to use the UniXcoder representation

        :return: list of names of the nodes in the networkx graph
        """
        # Turn AST while adding unixcoder node splits with their connections and create flattened representation
        # if root.has_error:
        #     raise ValueError(f'Tree has error at: {root}')

        # Split AST nodes into UniXcoder version, tokenize and split token nodes

        code_token = False
        nonescape_nodes = self.filter_instr_nodes(root)
        # Syntax node
        if len(nonescape_nodes) != 0:
            if unixcoder:
                base_names = [
                    f"AST#{root.type}#{suffix}" for suffix in ["Left", "Right"]
                ]
            else:
                base_names = [f"AST#{root.type}"]
        else:
            code_token = True
            # ? Assumes there is a space before each AST node
            # " " +
            base_name = Preprocessor.gen_name_from(root, index_to_code, skip_first=skip_first)
            # Remove initial space if this is the first token
            if not G.graph["first_token"]:
                G.graph["first_token"] = True
                if base_name[0] == " ":
                    base_name = base_name[1:]

            # Break down code tokens using given tokenizer if provided.
            if self.tokenizer is not None:
                base_names = self.tokenizer.tokenize(base_name)
            else:
                base_names = [base_name]

            # Add type if types are set to be added
            if (
                self.add_types
                and root.type in self.KEEP_TYPES
                and f"AST#{root.type}" not in G
            ):
                add_to_dict(self.code_types, f"AST#{root.type}")
                G.add_node(
                    f"AST#{root.type}",  # enumerated node name
                    flat_name=f"AST#{root.type}",  # actual token in the text
                    order=-1,
                    code_token=False,
                    ast_type=root.type,
                    # Untested
                    start_point=(-1, -1),
                    end_point=(-1, -1),
                )

        # Enumerate all nodes so they will always be added to graph
        names = []
        # Add nodes to graph
        for i, name in enumerate(base_names):
            # Build personal dictionaries for syntax and code tokens
            add_to_dict(self.vocab if code_token else self.ast_vocab, name)
            # prefix_space = ' ' if code_token and i==0 else ''
            unique_name = Preprocessor.get_unique_name(name, G)
            names.append(unique_name)
            G.add_node(
                unique_name,  # enumerated node name
                flat_name=name,  # actual token in the text
                order=i + len(flattened),
                code_token=code_token,
                ast_type=root.type,
                start_point=root.start_point,
                end_point=root.end_point,
            )
            if self.add_types and code_token and root.type in self.KEEP_TYPES:
                # Connect code token to its type.
                G.add_edge(
                    f"AST#{root.type}",
                    unique_name,
                    y=self.edge_types["type"],
                    **Preprocessor.VIS_OPTIONS["type"],
                )

        if len(root.children) != 0:
            flattened.append(base_names[0])
        else:
            for i, base_name in enumerate(base_names):
                # prefix_space = ' ' if code_token and i==0 else ''
                flattened.append(base_name)

        # Iterate over children, also implement UniXcoder flattener
        for child in root.children:
            if child.type in self.ignored_nodes() or "comment" in child.type:
                continue
            child_names = self._ts2nx(child, index_to_code, G, flattened, skip_first)
            # Fully connect splitted leafs (tokens) and AST nodes
            for name in names:
                # Connect all syntax nodes
                for child_name in child_names:
                    G.add_edge(
                        name,
                        child_name,
                        y=self.edge_types["parent"],
                        **Preprocessor.VIS_OPTIONS["parent"],
                    )

        if len(root.children) != 0 and unixcoder:
            G.nodes[names[1]]["order"] = len(flattened)
            flattened.append(base_names[1])
        return names


    def get_idx_code(self, node, index_to_code, skip_first=False):
        key = (
            (
                node.start_point[0] - (1 if skip_first else 0),
                node.start_point[1],
            ),
            (
                node.end_point[0] - (1 if skip_first else 0),
                node.end_point[1],
            )
        )
        if key not in index_to_code:
            key = (
                (
                    node.start_point[0] - (1 if skip_first else 0),
                    node.start_point[1] - 1,
                ),
                (
                    node.end_point[0] - (1 if skip_first else 0),
                    node.end_point[1],
                )
            )
        idx, code = index_to_code[key]
        return idx, code


    def tree_to_variable_index(self, root_node, index_to_code, skip_first=False):
        nonescape_nodes = self.filter_instr_nodes(root_node)
        if (
            len(nonescape_nodes) == 0 # or root_node.type == "string"
        ) and "comment" not in root_node.type:
            # index=(root_node.start_point,root_node.end_point)
            java_skip = (1 if skip_first else 0)
            index = (
                (
                    root_node.start_point[0] - java_skip,
                    root_node.start_point[1],
                ),
                (
                    root_node.end_point[0] - java_skip,
                    root_node.end_point[1],
                ),
            )
            if index not in index_to_code:
                index = (
                    (
                        root_node.start_point[0] - java_skip,
                        root_node.start_point[1] - 1,
                    ),
                    (
                        root_node.end_point[0] - java_skip,
                        root_node.end_point[1],
                    ),
                )
            return [index]
        else:
            code_tokens = []
            for child in root_node.children:
                code_tokens += self.tree_to_variable_index(
                    child, index_to_code, skip_first
                )
            return code_tokens
