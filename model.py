import sys

import torch
import random
import torch.nn as nn
import numpy as np
import sklearn
import torch.nn.functional as F
import dgl
import numpy as np
from utility.LightGCNLayer import LightGCNLayer
from utility.DotProductAtt import DotGatConv
import dgl.function as fn
import torchmetrics
# from kmeans_pytorch import kmeans as torch_kmeans
from sklearn.decomposition import PCA

'''TODO: construct prostive and negative item-bT_idx/cpr_idx bipartite graph for bpr loss'''


def construct_user_item_bigraph(graph):
    return graph.node_type_subgraph(['user', 'item'])


def construct_item_related_bigraph(graph, node_type='bT_idx'):
    return graph.node_type_subgraph([node_type, 'item'])


def construct_user_related_bigraph(graph, node_type='age'):
    return graph.node_type_subgraph([node_type, 'user'])


def construct_negative_item_graph(graph, k, device, node_type='bT_idx'):
    edge_dict = {'bT_idx': ['bi', 'ib'], 'cpr_idx': ['pi', 'ip']}
    user_item_src, user_item_dst = graph.edges(etype=edge_dict[node_type][0])
    neg_src = user_item_src.repeat_interleave(k)
    n_neg_src = len(user_item_src)
    neg_dst = torch.randint(0, graph.num_nodes(ntype='item'), (n_neg_src * k,)).to(device)
    data_dict = {
        (node_type, edge_dict[node_type][0], 'item'): (neg_src, neg_dst),
        ('item', edge_dict[node_type][1], node_type): (neg_dst, neg_src)
    }
    num_dict = {
        node_type: graph.num_nodes(ntype=node_type), 'item': graph.num_nodes(ntype='item'),
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


def construct_negative_user_graph(graph, k, device, node_type='age'):
    edge_dict = {'age': ['au', 'ua'], 'job': ['ju', 'uj']}
    user_item_src, user_item_dst = graph.edges(etype=edge_dict[node_type][0])
    neg_src = user_item_src.repeat_interleave(k)
    n_neg_src = len(user_item_src)
    neg_dst = torch.randint(0, graph.num_nodes(ntype='user'), (n_neg_src * k,)).to(device)
    data_dict = {
        (node_type, edge_dict[node_type][0], 'user'): (neg_src, neg_dst),
        ('user', edge_dict[node_type][1], node_type): (neg_dst, neg_src)
    }
    num_dict = {
        node_type: graph.num_nodes(ntype=node_type), 'user': graph.num_nodes(ntype='user'),
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


def arr_remove_idx(A, del_idx):  # return a 1-D array
    # print(A)
    # print(del_idx)
    mask = np.ones(A.shape, dtype=bool)
    mask[range(A.shape[0]), del_idx] = False
    output = A[mask]
    return output


def construct_negative_item_graph_c2ep(graph, n_type, device, node_type='cate'):  # n_cate = 12
    edge_dict = {'cate': ['ci', 'ic'], 'rate': ['ri', 'ir']}
    range_list = {'cate': np.arange(0, n_type, 1), 'rate': np.arange(0, n_type, 1)}
    user_item_src, user_item_dst = graph.edges(etype=edge_dict[node_type][1])
    # print(user_item_src.type())
    # print(user_item_dst[:40])
    neg_src = user_item_src.repeat_interleave(range_list[node_type].shape[0] - 1)
    neg_dst = np.array([range_list[node_type], ] * len(user_item_src))
    neg_dst = torch.from_numpy(arr_remove_idx(neg_dst, np.array(user_item_dst.cpu()))).to(device)
    data_dict = {
        (node_type, edge_dict[node_type][0], 'item'): (neg_dst, neg_src),
        ('item', edge_dict[node_type][1], node_type): (neg_src, neg_dst)
    }
    num_dict = {
        node_type: graph.num_nodes(ntype=node_type), 'item': graph.num_nodes(ntype='item'),
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


def construct_negative_user_graph_c2ep(graph, n_type, device, node_type='age'):  # n_cate = 12
    edge_dict = {'age': ['au', 'ua'], 'job': ['ju', 'uj']}
    range_list = {'age': np.arange(0, n_type, 1), 'job': np.arange(0, n_type, 1)}
    user_item_src, user_item_dst = graph.edges(etype=edge_dict[node_type][1])
    # print(user_item_src.type())
    # print(user_item_dst[:40])
    neg_src = user_item_src.repeat_interleave(range_list[node_type].shape[0] - 1)
    neg_dst = np.array([range_list[node_type], ] * len(user_item_src))
    neg_dst = torch.from_numpy(arr_remove_idx(neg_dst, np.array(user_item_dst.cpu()))).to(device)
    data_dict = {
        (node_type, edge_dict[node_type][0], 'user'): (neg_dst, neg_src),
        ('user', edge_dict[node_type][1], node_type): (neg_src, neg_dst)
    }
    num_dict = {
        node_type: graph.num_nodes(ntype=node_type), 'user': graph.num_nodes(ntype='user'),
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


def construct_negative_graph(graph, k, device):
    user_item_src, user_item_dst = graph.edges(etype='ui')
    neg_src = user_item_src.repeat_interleave(k)
    n_neg_src = len(user_item_src)
    neg_dst = torch.randint(0, graph.num_nodes(ntype='item'), (n_neg_src * k,)).to(device)
    data_dict = {
        ('user', 'ui', 'item'): (neg_src, neg_dst),
        ('item', 'iu', 'user'): (neg_dst, neg_src),
        # ('neg_user', 'ui', 'side'): (user_side_src, user_side_dst),
        # ('side', 'iu', 'neg_user'): (user_side_dst, user_side_src)
    }
    num_dict = {
        'user': graph.num_nodes(ntype='user'), 'item': graph.num_nodes(ntype='item'),
        # 'side': graph.num_nodes(ntype='side')
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']

    def alignment_forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                if etype == ('item', 'iu', 'user'):
                    continue
                edge_subgraph.apply_edges(
                    dgl.function.u_sub_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class HGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)
        self.edgedrop = dgl.transforms.DropEdge(0.2)
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
                # rst = torch.mul(F.sigmoid(edge_w), rst)
                # rst = F.softmax(torch.mul(edge_w, rst))
                # rst = torch.mul(F.leaky_relu(edge_w), rst)
                # rst = torch.mul(F.relu(edge_w), rst)
                # rst = torch.mul(edge_w, rst)
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


def prompt_cat(node_embedding, prompt_embedding, n_node):
    return torch.cat((prompt_embedding.expand(n_node, -1), node_embedding), dim=1)


class LightGCN(nn.Module):
    def __init__(self, args, graph, item_fc=False):
        super().__init__()
        self.layer_num = args.layer_num
        self.hid_dim = args.embed_size
        self.neg_samples = args.neg_samples
        self.decay = eval(args.regs)[0]
        self.item_text = args.item_text
        self.item_fc = item_fc if self.item_text == False else True
        self.build_model()
        self.user_embedding = torch.nn.Parameter(torch.randn(graph.nodes('user').shape[0], self.hid_dim))
        if self.item_text:
            self.item_embedding = graph.nodes['item'].data['tx']
            self.fc = nn.Linear(self.item_embedding.shape[1], self.hid_dim)
        else:
            self.item_embedding = torch.nn.Parameter(torch.randn(graph.nodes('item').shape[0], self.hid_dim))
            self.fc = nn.Linear(self.item_embedding.shape[1], self.hid_dim)
            torch.nn.init.normal_(self.item_embedding, std=0.1)

        # torch.nn.init.normal_(self.item_embedding, std=0.1)
        torch.nn.init.normal_(self.user_embedding, std=0.1)
        self.pred = ScorePredictor()
        self.node_features = {'user': self.user_embedding, 'item': self.item_embedding}

    def forward(self, graph):
        h = self.node_features
        if h['item'].shape[1] != self.hid_dim:
            self.item_embedding = self.fc(h['item'])
            h = {'user': self.user_embedding, 'item': self.item_embedding}
        else:
            if self.item_fc:
                fc_item_embedding = self.item_embedding
            else:
                fc_item_embedding = self.fc(h['item'])
            h = {'user': self.user_embedding, 'item': fc_item_embedding}

        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        for layer in self.lightGCNlayers:
            h_item = layer(graph, h, ('user', 'ui', 'item'))
            h_user = layer(graph, h, ('item', 'iu', 'user'))
            h = {'user': h_user, 'item': h_item}
            # print('in loop',h['user'].shape,h['item'].shape)
            user_embed.append(h_user)
            item_embed.append(h_item)
        user_embed_2 = torch.mean(torch.stack(user_embed, dim=0), dim=0)
        item_embed_2 = torch.mean(torch.stack(item_embed, dim=0), dim=0)
        h = {'user': user_embed_2, 'item': item_embed_2}
        return h

    def build_model(self):
        self.lightGCNlayers = nn.ModuleList()
        self.LGCNlayer = LightGCNLayer()
        for idx in range(self.layer_num):
            h2h = self.LGCNlayer
            self.lightGCNlayers.append(h2h)

    def create_bpr_loss(self, pos_g, neg_g, h, users_non_induct=None, loss_type='bpr'):
        sub_fig_feature = {'user': h['user'], 'item': h['item']}
        pos_score = self.pred(pos_g, sub_fig_feature)
        # print(pos_score)
        neg_score = self.pred(neg_g, sub_fig_feature)
        # print(neg_score)
        # sys.exit()
        pos_score = pos_score[('user', 'ui', 'item')].repeat_interleave(self.neg_samples, dim=0)
        neg_score = neg_score[('user', 'ui', 'item')]
        mf_loss = nn.Softplus()(neg_score - pos_score)
        mf_loss = mf_loss.mean()
        if users_non_induct != None:
            regularizer = (1 / 2) * (self.user_embedding[users_non_induct].norm(2).pow(2) +
                                     self.item_embedding.norm(2).pow(2))
        else:
            regularizer = (1 / 2) * (self.user_embedding.norm(2).pow(2) +
                                     self.item_embedding.norm(2).pow(2))
        emb_loss = self.decay * regularizer
        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    def hccf_create_ssl_loss_user(self, ini_h_user_v1, ini_h_user_v2, ssl_temp=0.1, ssl_reg=1e-6, k=2, reg=False):
        '''k=4 for goodreads3, k=2 for amazon/goodreads'''
        ssl_loss, regularizer = 0, 0
        ini_h_user_v1 = torch.nn.functional.normalize(ini_h_user_v1, p=2, dim=1)
        ini_h_user_v2 = torch.nn.functional.normalize(ini_h_user_v2, p=2, dim=1)
        for i in range(0, k):
            h_user_v1 = ini_h_user_v1.split(ini_h_user_v1.shape[0] // k)[i]
            h_user_v2 = ini_h_user_v2.split(ini_h_user_v2.shape[0] // k)[i]
            pos_score = torch.sum(torch.mul(h_user_v1, h_user_v2), dim=1)
            neg_score = torch.matmul(h_user_v1, h_user_v2.T)
            pos_score = torch.exp(pos_score / ssl_temp)
            neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
            ssl_loss += -torch.sum(torch.log(pos_score / neg_score))
        if reg:
            regularizer = reg * (1 / 2) * (ini_h_user_v1.norm(2).pow(2) +
                                           ini_h_user_v2.norm(2).pow(2))
        return ssl_reg * ssl_loss + regularizer

    def create_auloss(self, pos_g, h):
        # regularizer = (1 / 2) * (h['user'].norm(2).pow(2) + h['item'].norm(2).pow(2))
        pos_score = self.pred.alignment_forward(pos_g, h)
        auloss = pos_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2).mean()
        # emb_loss = self.decay * regularizer
        emb_loss = 0
        bpr_loss = auloss + emb_loss
        return bpr_loss, auloss, emb_loss

    def create_alignment_loss_batch(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def create_uniformity_loss_batch(self, x):
        x = F.normalize(x, dim=-1)
        loss = torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
        # print(loss)
        return 0.05 * loss

    def uniformity_loss(self, h_user, k=16, idx=0):
        x = h_user.split(h_user.shape[0] // k)[idx]
        x = F.normalize(x, dim=-1)
        uniformity_loss = torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
        return self.decay * uniformity_loss

class PromptGCN(LightGCN):
    def __init__(self, args, graph, device, fc_w=None, pretrain_user_embedding=None, edge_w=None, whitening_w=None,
                 finetuned_item_embedding=None, finetuned_user_embedding=None, item_split=None,
                 pretrain_cate_prompt=None, downstream_cate_prompt=None, pretrain=False):

        # print(graph.nodes['item'].data['tx'])
        super().__init__(args, graph)
        self.pretrain = pretrain
        self.prompt_bytask = args.prompt_bytask
        self.whitening = args.whitening
        self.device = device
        self.cate_embeddings = None
        self.norm_2 = args.norm_2
        self.fuse_decay = args.fuse_decay
        self.fuse_decay_user = args.fuse_decay_user
        self.ssl = args.ssl
        self.item_embedding = torch.nn.Parameter(torch.randn(graph.nodes('item').shape[0], self.hid_dim))
        self.interval = []
        self.pretrain_cate_prompt = pretrain_cate_prompt
        self.downstream_cate_prompt = downstream_cate_prompt
        # print('downstream_cate_prompt:',downstream_cate_prompt)
        if self.item_text == 1:
            self.item_text_embedding = graph.nodes['item'].data['tx'].clone().detach().requires_grad_(False)
            self.fc = nn.Linear(self.item_text_embedding.shape[1], self.hid_dim)
        elif self.item_text == 2:
            self.item_text_embedding = torch.nn.Parameter(torch.randn(graph.nodes('item').shape[0], self.hid_dim))
        '''the position of initialization will effect performance even with the same random seed'''
        torch.nn.init.normal_(self.item_embedding, std=0.1)
        if finetuned_item_embedding != None:
            self.item_embedding = nn.Parameter(finetuned_item_embedding)
        if finetuned_user_embedding != None:
            self.user_embedding.data[:finetuned_user_embedding.shape[0], :] = finetuned_user_embedding
        if args.edge_prompt:
            if edge_w != None:
                self.edge_w = edge_w.clone().detach().requires_grad_(True)
                # print(self.edge_w.requires_grad)
                # sys.exit()
            else:
                self.edge_w = torch.nn.Parameter(torch.randn(graph.nodes('user').shape[0], 1), requires_grad=True)
                torch.nn.init.normal_(self.edge_w, mean=1, std=0.5)
        else:
            self.edge_w = None
        if self.downstream_cate_prompt != None:
            self.cate_linear = nn.Linear(self.downstream_cate_prompt.shape[1] + self.item_text_embedding.shape[1],
                                         self.hid_dim)
            self.item_cate_embeddings = torch.cat((self.downstream_cate_prompt.repeat(self.item_text_embedding.shape[0],
                                                                                      1).to(device),
                                                   self.item_text_embedding), dim=1)
            print('category prompt in downstream tasks used with shape:', self.item_cate_embeddings.shape)


        else:
            if item_split != None:
                if pretrain_cate_prompt != None:
                    # print(cate_prompt.shape,self.item_text_embedding.shape)
                    '''choose! 1 for pre-cae 2 for post-cat'''
                    # self.cate_linear = nn.Linear(pretrain_cate_prompt.shape[1] + self.item_text_embedding.shape[1], self.hid_dim) #1
                    self.cate_linear = nn.Linear(self.hid_dim+pretrain_cate_prompt.shape[1], self.hid_dim) #2
                    cate_embeddings = []
                    for emb, count in zip(pretrain_cate_prompt, item_split):
                        cate_embeddings.extend([emb] * count)
                    self.cate_embeddings = torch.stack(cate_embeddings, dim=0).to(device)
                    self.item_cate_embeddings = torch.cat((self.cate_embeddings, self.item_text_embedding), dim=1)
                    print('category prompt in pretraining tasks used with shape:', self.item_cate_embeddings.shape)
                else:
                    self.linear_list = []
                    pos_begin = 0
                    pos_end = 0
                    for i in item_split:
                        pos_end += i
                        '''decide linear layer or MLP, also change the pretrain.py'''
                        # print('Here')
                        self.linear_list.append(
                            nn.Linear(self.item_text_embedding.shape[1], self.hid_dim, bias=True).to(device)
                            # nn.Sequential(
                            # nn.Linear(self.item_text_embedding.shape[1], self.hid_dim, bias=True),
                            # nn.ReLU(),
                            # nn.Dropout(p=0.2)
                            # ).to(device)
                        )
                        self.interval.append((pos_begin, pos_end))
                        pos_begin = pos_end
        if pretrain_user_embedding != None:
            self.user_embedding = nn.Parameter(pretrain_user_embedding)
            if args.user_pretrain == 2:  # frozen pretrained user embeddings
                self.user_embedding.requires_grad = False
            print('Pretrained users loaded.')
        if self.whitening == 1:
            num_g = 3
            text_d = self.item_text_embedding.shape[1]
            self.whitening_w1 = nn.Linear(text_d, self.hid_dim, bias=False)
            self.whitening_bias = torch.nn.Parameter(torch.randn(1, text_d), requires_grad=True)
            self.whitening_w2 = nn.Linear(text_d, num_g, bias=False)
            self.whitening_w3 = nn.Linear(text_d, num_g, bias=False)
            if pretrain == False:
                whitening_w1, whitening_w2, whitening_w3, self.whitening_bias = whitening_w
                self.whitening_w1.load_state_dict(whitening_w1)
                self.whitening_w2.load_state_dict(whitening_w2)
                self.whitening_w3.load_state_dict(whitening_w3)
                print('load whitening weights.')
        elif self.whitening == 2:
            num_cluster = 256
            pca = PCA(n_components=num_cluster)
            X = self.item_text_embedding.cpu()
            pca.fit(X)
            self.item_text_embedding = torch.from_numpy(pca.transform(X)).type(torch.cuda.FloatTensor).to(device)
            self.fc = nn.Linear(self.item_text_embedding.shape[1], self.hid_dim)
            # print(self.item_text_embedding.shape)
        if fc_w:
            self.fc.load_state_dict(fc_w)
            print('Pretrained text prompt loaded.')
            if args.text_prompt == 2:
                print('freeze text prompt')
                self.fc.weight.requires_grad = False
                self.fc.bias.requires_grad = False
        if self.item_text == 2:
            print('Cascaded PLM-GNN')
            del self.item_embedding
            self.item_embedding = graph.nodes['item'].data['tx'].clone().detach().requires_grad_(True)
        self.node_features = {'user': self.user_embedding, 'item': self.item_embedding}

        if args.hgcn == 2:
            self.build_lightgcn_model()
        elif args.hgcn == 3:
            self.dhcf_linear_user = nn.Linear(self.hid_dim, self.hid_dim)
            self.dhcf_linear_item = nn.Linear(self.hid_dim, self.hid_dim)
            self.dhcf_linear_joint = nn.Linear(self.hid_dim, self.hid_dim)

    def build_model(self):
        print('build HGCN model')
        self.layers = nn.ModuleList()
        self.HGCNLayer = HGCNLayer()
        # self.fc = nn.Linear(384, self.hid_dim)
        for idx in range(self.layer_num):
            h2h = self.HGCNLayer
            self.layers.append(h2h)

    def build_lightgcn_model(self):
        print('build LGCN model')
        self.lightGCNlayers = nn.ModuleList()
        self.LGCNlayer = LightGCNLayer()
        for idx in range(self.layer_num):
            h2h = self.LGCNlayer
            self.lightGCNlayers.append(h2h)

    def forward(self, graph):
        # print(self.node_features['user'].shape)
        if self.interval == []:
            if self.pretrain_cate_prompt == None:
                if self.downstream_cate_prompt != None:
                    item_text_embedding = self.cate_linear(self.item_cate_embeddings)
                    # print('downstream forward:',item_text_embedding.shape)
                else:
                    # print('this branch')
                    if self.whitening == 1:
                        x_i = self.whitening_w1(self.item_text_embedding - self.whitening_bias)
                        sigma = F.normalize(F.softplus(self.whitening_w3(self.item_text_embedding)), dim=0)
                        g = F.softmax(self.whitening_w2(self.item_text_embedding) + sigma, dim=0)
                        item_text_embedding = torch.sum(torch.mul(g.unsqueeze(2), x_i.unsqueeze(1)), dim=1)
                    else:
                        '''the ONLY VALID branch'''
                        if self.item_text == 1:
                            item_text_embedding = self.fc(self.item_text_embedding)
                        elif self.item_text == 2:
                            item_text_embedding = self.item_text_embedding
                        else:
                            item_text_embedding = None
                        # print(self.item_text_embedding)
                        # sys.exit()
                    # print('downstream forward 2:',item_text_embedding.shape)
            else:
                '''Choose pre-concatenate here or post-concatenate later'''
                # print('Here!')
                item_text_embedding = self.fc(self.item_text_embedding) if self.item_text else None
                # item_text_embedding = self.cate_linear(self.item_cate_embeddings)
                # print('pretrain forward:',item_text_embedding.shape)
        else:
            transformed_list = []
            for idx in range(len(self.interval)):
                cur_item_split = self.item_text_embedding[self.interval[idx][0]:self.interval[idx][1]]
                transformed_part = self.linear_list[idx](cur_item_split)
                transformed_list.append(transformed_part)
                item_text_embedding = torch.cat(transformed_list, dim=0)
        h = self.node_features
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]

        if self.item_text == 2:
            self.item_embedding_cascaded = self.fc(self.item_embedding)
            h = {'user': self.user_embedding, 'item': self.item_embedding_cascaded}
            item_embed = [self.item_embedding_cascaded]
        if self.ssl:
            user_embed_v2 = [self.user_embedding]

        for layer in self.layers:
            # h_item = layer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), self.norm_2, edge_w=self.edge_w,
            #                alpha=self.fuse_decay_user, pretrained_feature=h['user'])
            h_user = layer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), self.norm_2, alpha=self.fuse_decay,
                           pretrained_feature=item_text_embedding)
            h_item = layer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), self.norm_2, edge_w=self.edge_w,
                           alpha=self.fuse_decay_user, pretrained_feature=h_user)
            h = {'user': h_user, 'item': h_item}
            user_embed.append(h_user)
            item_embed.append(h_item)

            if self.ssl:
                mask_user = torch.randn(graph.nodes('item').shape[0], 1).to(self.device)
                h_user_v2 = layer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), self.norm_2,
                                  edge_w=mask_user)
                user_embed_v2.append(h_user_v2)

        user_embed = torch.mean(torch.stack(user_embed, dim=0), dim=0)
        item_embed = torch.mean(torch.stack(item_embed, dim=0), dim=0)

        '''Choose post-concatenate here or pre-concatenate before'''
        if self.cate_embeddings != None:
            # print('There!')
            item_embed = self.cate_linear(torch.cat((item_embed,self.cate_embeddings),dim=1))

        h = {'user': user_embed, 'item': item_embed}
        if self.ssl:
            user_embed_v2 = torch.mean(torch.stack(user_embed_v2, dim=0), dim=0)
            self.h_user_v1 = user_embed
            self.h_user_v2 = user_embed_v2
        return h


    def create_ssl_loss_user(self, ssl_temp):
        # ssl_temp = 0.1
        h_user_v1 = torch.nn.functional.normalize(self.h_user_v1, p=2, dim=1)
        h_user_v2 = torch.nn.functional.normalize(self.h_user_v2, p=2, dim=1)

        # pos_score = torch.sum(torch.mul(h_user_v1, h_user_v2), dim=1)
        # pos_score = torch.exp(pos_score / ssl_temp)

        pos_score = 1 / ssl_temp

        neg_score = torch.matmul(h_user_v1, h_user_v2.T)
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return self.decay * ssl_loss

    def create_ssl_loss_item(self, ssl_temp):
        ssl_temp = 0.1
        h_item_v1 = torch.nn.functional.normalize(self.h_item_v1, p=2, dim=1)
        h_item_v2 = torch.nn.functional.normalize(self.h_item_v2, p=2, dim=1)
        neg_score = torch.matmul(h_item_v1, h_item_v2.T)
        pos_score = 1 / ssl_temp
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def create_ssl_loss_batched_item(self, ssl_temp, k=4):
        ssl_temp = 0.1
        idx = random.randint(0, k - 1)
        h_item_v1 = self.h_item_v1.split(self.h_item_v1.shape[0] // k + 1)[idx]
        h_item_v2 = self.h_item_v2.split(self.h_item_v2.shape[0] // k + 1)[idx]
        h_item_v1 = torch.nn.functional.normalize(h_item_v1, p=2, dim=1)
        h_item_v2 = torch.nn.functional.normalize(h_item_v2, p=2, dim=1)
        neg_score = torch.matmul(h_item_v1, h_item_v2.T)
        pos_score = 1 / ssl_temp
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

