from dataclasses import dataclass
from copy import deepcopy

from ..models import GNN

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

from adapters.configuration import ParBnConfig
from adapters.methods.modeling import Adapter, ParallelAdapter
from adapters.methods.adapter_layer_base import AdapterLayerBase
from adapters import (
    ForwardContext,
    InvertibleAdaptersMixin,
    InvertibleAdaptersWrapperMixin,
)


def apply_to_adapter_layers(self, fn):
    """
    Applies a function to all adapter layers of the model.
    """
    for i, layer in self.iter_layers():
        for module in layer.modules():
            if isinstance(module, AdapterLayerBase):
                if i in self.target_layers:
                    fn(i, module)


def add_adapter(model, adapter_name, adapter_config, target_layers, core, is_encoder=True, **kwargs):
    # Override iterator so it only goes over target adapters
    model.base_model.target_layers = target_layers
    setattr(
        model.base_model.__class__,
        "apply_to_adapter_layers",
        apply_to_adapter_layers,
    )

    # Add parallel adapters to the model
    model.add_adapter(adapter_name, config=adapter_config)

    # Overwrite parallel adapters so that they are our modules with weights
    all_adapters = []
    for i, (l_i, layer) in enumerate(model.iter_layers()):
        for module in layer.modules():
            if isinstance(module, AdapterLayerBase):
                # Get initialized adapter
                target_adapter = module.get_adapter(adapter_name)
                if target_adapter is not None:
                    # Cast adapter to Graph adapter and set NN with provided core
                    GraphAdapter.overwrite_bottleneck(
                        target_adapter, core, adapter_config
                    )
                    target_adapter.adv_ptr = False
                    target_adapter.is_encoder = is_encoder
                    for k, v in kwargs.items():
                        setattr(target_adapter, k, v)
                    all_adapters.append(target_adapter)
    return all_adapters


def freeze_adapters(model, adapter_name, train_embeddings=False):
    """Built upon adapters.ModelAdaptersMixin.train_adapter function to simply
    freeze adapters and set model to train instead!
    """
    model.train()
    model.freeze_model(False)

    model.apply_to_adapter_layers(lambda _, layer: layer.freeze_adapter(adapter_name))
    model.apply_to_basemodel_childs(lambda _, child: child.freeze_adapter(adapter_name))

    if adapter_name in model.base_model.shared_parameters:
        for param in model.base_model.shared_parameters[adapter_name].values():
            param.requires_grad = False

    if isinstance(model, InvertibleAdaptersMixin) or isinstance(
        model, InvertibleAdaptersWrapperMixin
    ):
        if adapter_name in model.invertible_adapters:
            for param in model.invertible_adapters[adapter_name].parameters():
                param.requires_grad = False
    # We assume adapters are already active
    # model.set_active_adapters(adapter_setup)
    if train_embeddings:
        model.get_input_embeddings().train()
    else:
        model.get_input_embeddings().eval()
    model.get_input_embeddings().weight.requires_grad = not train_embeddings


@dataclass(eq=False)
class GeometricAdapterConfig(ParBnConfig):
    """
    Graph based adapter config
    """

    # ! GNN parameters, fix parameters
    gnn_d_model: float = 5
    gnn_d_ff: float = 5
    gnn_num_relations: float = 2
    gnn_root_weight: bool = True
    gnn_dropout: float = 0.5


class GraphAdapter(ParallelAdapter):
    @classmethod
    def overwrite_bottleneck(
        cls, adapter: Adapter, core: GNN, config: GeometricAdapterConfig
    ):
        adapter.__class__ = cls

        # Bottleneck adapter parts from Adapterhub implementation
        del adapter.adapter_down
        del adapter.adapter_up
        # set base layer
        adapter.gnn_adapter = deepcopy(core)
        adapter.config = config
        # For logging purposes
        adapter.start_offset_func = "start_offset + 1 + (2 if shift else 0)"

    def pre_forward(
        self,
        hidden_states,
        input_tensor,
        layer_norm,
        fusion_config=None,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration.

        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        # In case of parallel adapter, return the input tensor as hidden states
        query = None
        if fusion_config is not None:
            query = input_tensor
        return input_tensor, query, input_tensor

    def forward(self, x, residual_input, output_gating=False):
        parameters = ForwardContext.get_context().shared_parameters[self.name]
        # Get shared graph batch
        graph_batch = parameters["graph_batch"]
        pad_emb = parameters["pad_emb"]
        # This offset is from our preparation in dataset
        start_offset = self.start_offset["input" if self.is_encoder else "output"]
        # We are using HF forward and generate functions with bos_token_id and decoder_token_id values
        # These are set in #task.py configure_special_tokens function
        # So we need to skip these tokens in the beginning
        # e.g. CodeT5:
        #      <pad> <s> int function ...
        #      ^     ^
        #      dti   dataset
        # dti: Added by HF that is decoder_token_id
        # dataset: Added by us in data preparation
        # So we need to skip 2 tokens in the beginning
        shift = self.is_train and (self.model_name in ["t5"])
        # shift = False
        # Do it this way so we can report the equation
        self.start_offset_func = "start_offset + 1 - (0 if shift else 1)"
        start_offset = eval(self.start_offset_func)
        eos_offset = 1

        start_token = False
        if hasattr(self, "input_ids"):
            input_ids = self.input_ids
            if input_ids.shape[1] == 1 and input_ids[0][0] == 0:
                start_token = True

        # past_values are set as embeddings if layer mode
        if self.emb_source == "layer":
            if x.shape[1] == 1:  # Validation without beam, generation
                if start_offset <= self.cursor:
                    # Advance by pointer (over code tokens) in each graph to find next code token index
                    graph_idx = graph_batch.cptr[:-1] + self.cursor - start_offset
                    # Check valid updates: do not update next graphs (or further than token limit)
                    valid_mask = graph_idx < (graph_batch.cptr[1:] - graph_batch.extras)
                    # Get update idx in full graph location
                    update_idx = self.code_indices[graph_idx[valid_mask]]
                    # Update related only related tensors
                    # Condition check if we are in generative mode or training mode
                    self.past_values[update_idx] = x.squeeze()[valid_mask]
                    # Calculate the edge mask
                    edge_mask = graph_batch.edge_mask <= (self.cursor - start_offset)

                    # Here we push GNN with previous layer's hidden representation
                    up = self.gnn_adapter.attached_forward(
                        self.past_values, graph_batch, edge_mask=edge_mask
                    )
            else:
                # This is one pass, from encoder side
                # Or this is called on decoder side when learning
                # Get code indices into block tensor with padding
                splits = torch.split(self.code_indices, graph_batch.code_len.tolist())
                # Cursor is set to be the max generation length in encoder adapters.
                # This precut is for memory efficiency to apply before pad_sequence
                splits = [s[:self.cursor] if len(s) > self.cursor else s for s in splits]
                # Pad current sequence to match the shape of x
                graph_mask = pad_sequence(splits, True, -100)[
                    :, : x.shape[1] - start_offset - eos_offset
                ]
                # Get the corresponding mask for the transformer
                seq_mask = graph_mask != -100
                # Get the mask for graph back, clipped
                graph_mask = graph_mask[seq_mask]

                # The the features from previous layers for GNN to process
                self.past_values[graph_mask] = x[
                    :,
                    start_offset : min(
                        seq_mask.shape[1] + start_offset, x.shape[1] - eos_offset
                    ),
                    :,
                ][seq_mask]
                # cursor set to be max_source_length by main experiment
                edge_mask = graph_batch.edge_mask <= (self.cursor - start_offset)
                # Forward GNN to get features
                up = self.gnn_adapter.attached_forward(self.past_values, graph_batch, edge_mask=edge_mask)
        else:
            if x.shape[1] == 1:
                if self.start_offset <= self.cursor:
                    # Calculate the edge mask
                    edge_mask = graph_batch.edge_mask <= self.cursor
            else:
                edge_mask = None

            if self.pl_feed == "input":
                edge_mask = None
            # Here GNN uses different source of features
            up = self.gnn_adapter.attached_forward(
                None, graph_batch, edge_mask=edge_mask
            )

        # ========== FEATURE ALIGNMENT ===========
        # This part is for unpadded generation adaptation to
        # transformers generation.

        # output = torch.zeros_like(x)
        # output[valid_mask] = up[update_idx][:, None, :]
        if x.shape[1] == 1:  # Validation without beam, generation
            # Not sure if it is ineffective this way...
            output = torch.zeros_like(x)
            if start_offset <= self.cursor:
                output[valid_mask] = up[update_idx][:, None, :]
        else:
            # Break splits of code token features from GNN
            splits = torch.split(
                up[graph_batch.code_mask], graph_batch.code_len.tolist()
            )
            max_without_special = x.shape[1] - start_offset - eos_offset
            # Get individual graph lengths clips
            gnn_features_idx = torch.clamp(
                graph_batch.code_len, max=max_without_special
            )
            # Filter valid indices and concat with padding to match output shape
            all_gnn_features = pad_sequence(
                [split[:clmp] for clmp, split in zip(gnn_features_idx, splits)], True
            )
            # Get the GNN features, reshape it to match `x` 64 x 256 x 768
            # skip the starting offset positions
            # e.g. <pad>   <s>   int function ...  </s> <pad> <pad>
            #      start_offset  all_gnn_features  eos_offset
            #      0       0     all_gnn_features  0    0     0
            output = pad(
                all_gnn_features, (0, 0, start_offset, x.shape[1] - start_offset - all_gnn_features.shape[1])
            )

        # Update graph cursor
        self.cursor = self.cursor + 1
        if self.adv_ptr:
            graph_batch.gptr = graph_batch.gptr + 1

        output = output * self.scaling
        # changing the strength of GNN features
        if getattr(self, "scale_decay", None) is not None:
            output = output * self.scale_decay.factor
        up = output
        down = output

        # Default false
        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # Default false, apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up

    def post_forward(
        self, hidden_states, input_hidden_states, input_tensor, layer_norm
    ):
        """
        Performs computations after the forward pass of the adapter block(s). This e.g. includes applying the residual
        connection and layer norm if configured in this way.

        Args:
            hidden_states: The hidden states outputted by the adapter block(s).
            input_hidden_states: Residual connection before the adapter block(s).
            input_tensor: Residual connection before the Transformer FFN/ attention layer.
            layer_norm: Transformer LayerNorm.

        Returns:
            The modified hidden states.
        """
        hidden_states = hidden_states + input_hidden_states

        if self.original_ln_after:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        return hidden_states
