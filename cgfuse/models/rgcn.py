from . import GNN
from torch_geometric.nn import RGCNConv


class RGCN(GNN):
    supports_edge_weight = False
    supports_edge_attr = False
    supports_edge_type = True

    def init_conv(self, in_features, out_features):
        return RGCNConv(
            in_features,
            out_features,
            num_relations=self.args.num_relations,
            root_weight=self.args.root_weight
        )

    def layer_name(self):
        return "rgcn_conv"