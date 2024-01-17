import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

class LightGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype, norm_2 = -0.5):
        with graph.local_scope():
            src, _, dst = etype
            feat_src = h[src]
            feat_dst = h[dst]
            aggregate_fn = fn.copy_src('h', 'm')

            out_degrees = graph.out_degrees(etype=etype).float().clamp(min=1)
            norm_src = torch.pow(out_degrees, norm_2)

            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
            norm_src = torch.reshape(norm_src, shp_src)
            feat_src = feat_src * norm_src

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype)

            rst = graph.nodes[dst].data['h']
            in_degrees = graph.in_degrees(etype=etype).float().clamp(min=1)
            norm_dst = torch.pow(in_degrees, norm_2)
            shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm_dst, shp_dst)
            rst = rst * norm

            return rst
    def maxpool(self, graph, h, etype, norm_2 = -0.5):
        with graph.local_scope():
            src, _, dst = etype
            feat_src = h[src]
            aggregate_fn = fn.copy_src('h', 'm')

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, fn.max(msg='m', out='h'), etype=etype)

            rst = graph.nodes[dst].data['h']
            return rst
    def meanpool(self, graph, h, etype, norm_2 = -0.5):
        with graph.local_scope():
            src, _, dst = etype
            feat_src = h[src]
            aggregate_fn = fn.copy_src('h', 'm')

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, fn.mean(msg='m', out='h'), etype=etype)

            rst = graph.nodes[dst].data['h']
            return rst
class HGCNLayer_general(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype_list, norm_2=-1):
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')
            aggregate_fn_back = fn.copy_src('h_b', 'm_b')
            for etype in etype_list:
                etype_forward, _ = etype
                src, _, dst = etype_forward
                feat_src = h[src]
                feat_dst = h[dst]

                graph.nodes[src].data['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype_forward)

                rst = graph.nodes[dst].data['h']
                in_degrees = graph.in_degrees(etype=etype_forward).float().clamp(min=1)
                norm_dst = torch.pow(in_degrees, -1)
                shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm_dst, shp_dst)
                rst = rst * norm
                graph.nodes[dst].data['h_b'] = rst

            update_dict = {}
            in_degrees_b = None
            for etype in etype_list:
                _, etype_back = etype
                update_dict[etype_back] = (aggregate_fn_back, fn.sum(msg='m_b', out='h_b'))
                if in_degrees_b == None:
                    in_degrees_b = graph.in_degrees(etype=etype_back).float().clamp(min=1)
                else:
                    in_degrees_b += graph.in_degrees(etype=etype_back).float().clamp(min=1)
            graph.multi_update_all(update_dict, 'sum')
            bsrc = graph.nodes[src].data['h_b']

            norm_src = torch.pow(in_degrees_b, norm_2)
            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
            norm_src = torch.reshape(norm_src, shp_src)
            bsrc = bsrc * norm_src

            return bsrc, rst

class HGCNLayer_DHCF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)
        # self.dropout = nn.Dropout(p=0.01)

    def forward(self, graph, h, etype_forward, etype_back, norm_2=-1, alpha=0, pretrained_feature=None, edge_w=None):
        with graph.local_scope():
            src, _, dst = etype_forward
            feat_src = h[src]
            feat_dst = h[dst]
            aggregate_fn = fn.copy_src('h', 'm')
            aggregate_fn_back = fn.copy_src('h_b', 'm_b')

            graph.nodes[src].data['h'] = feat_src
            # graph.nodes[src].data['h'] = self.dropout(feat_src)
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype_forward)

            rst = graph.nodes[dst].data['h']
            in_degrees = graph.in_degrees(etype=etype_forward).float().clamp(min=1)
            norm_dst = torch.pow(in_degrees, -1)
            shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm_dst, shp_dst)

            if alpha != 0 and pretrained_feature != None:
                rst = rst + alpha * torch.tanh(pretrained_feature)
            if edge_w != None:
                rst = torch.mul(F.softmax(edge_w), rst)
            rst = rst * norm

            graph.nodes[dst].data['h_b'] = rst
            graph.update_all(aggregate_fn_back, fn.sum(msg='m_b', out='h_b'), etype=etype_back)
            bsrc = graph.nodes[src].data['h_b']

            in_degrees_b = graph.in_degrees(etype=etype_back).float().clamp(min=1)
            norm_src = torch.pow(in_degrees_b, norm_2)
            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
            norm_src = torch.reshape(norm_src, shp_src)
            bsrc = bsrc * norm_src

        return bsrc