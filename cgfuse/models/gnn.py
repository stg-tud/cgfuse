from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, MessagePassing
from transformers import AutoModel

from . import Model

class GNN(Model):
    # Customized implementation of # Parameters are gathered from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GraphSAGE
    # Some parts are gathered from that implementation
    supports_edge_weight = False
    supports_edge_attr = False
    supports_edge_type = False

    def __init__(self, args):
        super().__init__(args)

        pre_trained_model = AutoModel.from_pretrained(self.args.embeddings.model_name)
        emb_matrix = (
            pre_trained_model.get_input_embeddings().weight.detach().cpu().numpy()
        )

        self.embeddings = nn.Embedding(emb_matrix.shape[0], emb_matrix.shape[1])
        self.embeddings.weight.data.copy_(torch.from_numpy(emb_matrix))

        self.activation = getattr(F, args.activation)
        self.layer_norm = LayerNorm(self.args.in_features)
        self.dropout_gnn = nn.Dropout(self.args.dropout)

        # Predictor, this is for standalone training.
        self.wo = nn.Linear(
            self.args.out_features,
            self.args.vocab_size,
            bias=False,
        )

        self.init_blocks()

    def init_blocks(self):
        i = 0
        self.layers = []
        in_features = self.args.in_features
        for i, out_features in enumerate(self.args.intermediate):
            conv = self.init_conv(in_features, out_features)
            self.add_module(f"{self.layer_name()}{i}", conv)
            self.layers.append(conv)
            in_features = out_features
        conv = self.init_conv(in_features, self.args.out_features)
        self.add_module(f"{self.layer_name()}{i+1}", conv)
        self.layers.append(conv)

    def init_conv(self, in_features, out_features) -> MessagePassing:
        pass

    def layer_name(self):
        return ""

    def forward_layer(
        self, conv, x, edge_index=None, edge_weight=None, edge_attr=None, edge_type=None
    ):
        kwargs = {}
        if self.supports_edge_weight:
            kwargs["edge_weight"] = edge_weight
        if self.supports_edge_attr:
            kwargs["edge_attr"] = edge_attr
        if self.supports_edge_type:
            kwargs["edge_type"] = edge_type

        return conv(x, edge_index, **kwargs)

    def attached_forward(self, past_values, graphs, edge_mask=None):
        gnn_out = graphs.x if past_values is None else past_values
        if edge_mask == None:
            edge_index = graphs.edge_index
            edge_type = graphs.edge_type
        else:
            # Cut the edges to unseen nodes
            edge_index = graphs.edge_index[:, edge_mask]
            edge_type = graphs.edge_type[edge_mask]
        for layer in self.layers:
            # gnn_out = layer(gnn_out, graphs.edge_index, edge_type=graphs.edge_type)
            gnn_out = self.forward_layer(
                layer, gnn_out, edge_index, edge_type=edge_type
            )
            gnn_out = self.activation(gnn_out)

        gnn_out = self.layer_norm(gnn_out)
        gnn_out = self.dropout_gnn(gnn_out)

        return gnn_out

    def forward(self, graphs):
        gnn_out = self.embeddings(graphs.x)
        for layer in self.layers:
            # gnn_out = layer(gnn_out, graphs.edge_index, edge_type=graphs.edge_type)
            gnn_out = self.forward_layer(
                layer, gnn_out, graphs.edge_index, edge_type=graphs.edge_type
            )
            gnn_out = self.activation(gnn_out)

        # Regularize
        gnn_out = self.layer_norm(gnn_out)
        gnn_out = self.dropout_gnn(gnn_out)
        # Classify node
        gnn_out = self.wo(gnn_out)

        return gnn_out

    def strip(self):
        embeddings = self.embeddings
        wo = self.wo
        del self.wo
        del self.embeddings
        return wo, embeddings

    @staticmethod
    def extend_tokenizer(tokenizer, ast_types: Union[set, list] = None):
        """Special for Unixcoder!
        Update Unixcoder tokenizer to contain AST node vocab

        This extracts all <AST-Node>#Left and <AST-Node>#Right embeddings and gets the mean of the to have
        <AST-Node> embeddings. Left and Right embeddings are added as Unixcoder representation of ASTs and
        do not match with default representation.

        :param tokenizer: HF tokenizer, in this case specifically UnixCoder's.
        :param ast_types: List or set of AST node types
        :return: Dict of e.g. "AST#method_declaration": ["AST#method_declaration#Left", "AST#method_declaration#Right"]
        """
        if ast_types is None:
            ast_types = {}
            for token in tokenizer.vocab:
                # find AST# tokens
                if token.startswith("AST#"):
                    # keep string before last #
                    cut_token = "AST#" + token.split("#")[-2]
                    if cut_token in ast_types:
                        ast_types[cut_token].append(tokenizer.vocab[token])
                    else:
                        ast_types[cut_token] = [tokenizer.vocab[token]]

            # add the new tokens to the self.tokenizer
            tokenizer.add_tokens(list(ast_types.keys()))
        else:
            tokenizer.add_tokens(list(ast_types))

        return ast_types

    def extend_embeddings(self, tokenizer, ast_types: Union[set, list, dict]):
        """Special for Unixcoder!
        Update embeddings to match updated tokenizer

        extends embeddings matrix according to ast_types provided. This param is received from extend_tokenizer function.

        :param tokenizer: tokenizer to load embeddings for
        :param ast_types: Dict of e.g. "AST#method_declaration": ["AST#method_declaration#Left", "AST#method_declaration#Right"]
        :return: new tokens with their ids
        """
        new_embeddings = self.resize_token_embeddings(len(tokenizer))

        # Initialize new embeddings from previously seperated one if unixcoder
        # emb("AST#method_declaration"): mean([emb("AST#method_declaration#Left"), emb("AST#method_declaration#Right")])
        if isinstance(ast_types, dict):
            # freezing embedding values
            # ? new embeddings are not trained
            # new_embeddings.weight.requires_grad = False

            new_ast_types = {}
            for token, values in ast_types.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                new_ast_types[token] = token_id
                with torch.no_grad():
                    new_embeddings.weight.data[token_id] = new_embeddings.weight[values].mean(dim=0)
        else:
            # Random initialization
            new_ast_types = {k: None for k in ast_types}

        self.embeddings = new_embeddings

        # Resize the predictor to fit new vocab_size
        self.wo = nn.Linear(
            self.embeddings.weight.shape[1],
            self.embeddings.weight.shape[0],
            bias=False,
        )

        return new_ast_types
