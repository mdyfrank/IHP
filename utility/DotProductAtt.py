"""Torch modules for graph attention networks(GAT)."""
import sys

# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn
import dgl
from dgl import function as fn

from dgl.backend import edge_softmax as edge_softmax_internal
from dgl.backend  import edge_softmax_hetero as edge_softmax_hetero_internal
from dgl.backend import astype
from dgl.base import ALL, is_all

from dgl.base import DGLError
from dgl.utils import expand_as_pair


class DotGatConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 allow_zero_in_degree=False):
        super(DotGatConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, self._out_feats*self._num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=False)

    def forward(self, graph, feat, etype, get_attention=False):
        (src_type,edge_ui,dst_type) = etype

        graph = graph.local_var()
        graph = graph.edge_type_subgraph([edge_ui])

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            # print(feat_src.shape)
            # print(feat_dst.shape)
            # sys.exit()
        else:
            h_src = feat
            feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]

        # Assign features to nodes
        graph.nodes[src_type].data.update({'ft': feat_src})
        graph.nodes[dst_type].data.update({'ft': feat_dst})
        # graph.dstdata.update({'item': feat_dst})

        # sys.exit()
        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))
        # print(edge_softmax_half(graph, graph.edata['a']))
        # print(graph.edges(etype=edge_ui,form='eid'))
        # sys.exit()
        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax_half(graph, graph.edata['a']/ self._out_feats**0.5)
        # print(graph.edata['sa'].shape)
        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e('ft', 'sa', 'attn'), fn.sum('attn', 'agg_u'))


        # output results to the destination nodes
        rst = graph.dstdata['agg_u']
        # print(rst.shape)
        # sys.exit()
        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst

"""dgl edge_softmax operator module."""


__all__ = ['edge_softmax']


def edge_softmax_half(graph, logits, eids=ALL, norm_by='dst'):

    if not is_all(eids):
        eids = astype(eids, graph.idtype)
    if graph._graph.number_of_etypes() == 1:
        return edge_softmax_internal(graph._graph, logits,
                                     eids=eids, norm_by=norm_by)
    else:
        logits_list = [None] * graph._graph.number_of_etypes()
        for rel in graph.canonical_etypes:
            etid = graph.get_etype_id(rel)
            logits_list[etid] = logits[rel]
        logits_tuple = tuple(logits_list)
        score_tuple = edge_softmax_hetero_internal(graph._graph,
                                                   eids, norm_by, *logits_tuple)
        score = {}
        for rel in graph.canonical_etypes:
            etid = graph.get_etype_id(rel)
            score[rel] = score_tuple[etid]
        return score
