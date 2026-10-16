from . import GNN
from torch_geometric.nn import GINConv, MLP


class GIN(GNN):
    supports_edge_weight = False
    supports_edge_attr = False
    supports_edge_type = False

    def init_conv(self, in_features, out_features):
        # Parameters are gathered from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GraphSAGE
        mlp = MLP(
            [in_features, out_features, out_features],
            act="relu", # self.act,
            act_first=False, # self.act_first,
            norm=None, # self.norm,
            norm_kwargs=None # self.norm_kwargs,
        )
        return GINConv(mlp)

    def layer_name(self):
        return "gin_conv"