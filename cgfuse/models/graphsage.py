from . import GNN
from torch_geometric.nn import SAGEConv


class GraphSAGE(GNN):
    supports_edge_weight = False
    supports_edge_attr = False
    supports_edge_type = False

    def init_conv(self, in_features, out_features):
        return SAGEConv(
            in_features,
            out_features,
            root_weight=self.args.root_weight,
        )

    def layer_name(self):
        return "sage_conv"