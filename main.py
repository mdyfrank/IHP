import random
import sys

import torch
import torch.optim as optim
from utility.load_data import *
from utility.parser import *
from utility.batch_test import *
from utility.helper import log, early_stopping, random_batch_users, ensureDir, convert_dict_list, convert_list_str
from model import *
from time import time
from pretrain import *
from pretrain_baselines import *
import coldstart
import numpy as np
import dgl


def main(args):
    args = parse_args()
    # data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, dataset = args.dataset)
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    whitening_w = None
    fc_w, pretrain_user_embedding, edge_w, whitening_w = pretrain(args, device)
    users_to_test = list(data_generator.test_set.keys())
    g = data_generator.g
    print('pretrain completed')
    if args.hgcn == 3:
        new_g = torch.sparse.mm(torch.sparse.mm(g.adj(etype='ui'), g.adj(etype='iu')), g.adj(etype='ui'))
        src, dst = new_g.coalesce().indices()
        dhcf_g_data_dict = {
            ('user', 'ui', 'item'): (src, dst),
            ('item', 'iu', 'user'): (dst, src), }
        dhcf_g_num_dict = {
            'user': data_generator.n_users, 'item': data_generator.n_items
        }
        dhcf_g = dgl.heterograph(dhcf_g_data_dict, num_nodes_dict=dhcf_g_num_dict)
        if args.dataset == 'goodreads3':
            transform = dgl.DropEdge(p=0.8)
            dhcf_g = transform(dhcf_g)
        onetwohop_g = dgl.merge([g, dhcf_g])
        print('dhcf big graph', onetwohop_g)
    g = g.to(device)
    pos_g = construct_user_item_bigraph(g)
    if args.text_prompt == 0:
        fc_w = None
    if args.user_pretrain == 0:
        pretrain_user_embedding = None
    if args.edge_prompt == 0:
        edge_w = None
    if args.lightgcn:
        if args.inductive:
            model = GraphSage(args, g).to(device)
        else:
            model = LightGCN(args, g).to(device)
    elif args.gat:
        model = GraphFormer(args, g).to(device)
    else:
        downstream_cate_prompt = data_generator.downstream_cate_prompt.to(device) if args.prompt_bytask else None
        model = PromptGCN(args, g, device, fc_w, pretrain_user_embedding, edge_w, whitening_w,
                          downstream_cate_prompt=downstream_cate_prompt).to(device)
    t0 = time()
    cur_best_pre_0, stopping_step = 0, 0
    ssl_loss, emb_loss, mf_loss = 0, 0, 0
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    user_lr = args.user_lr if args.user_lr != -1 else args.lr
    param_groups = [
        {'params': model.user_embedding, 'lr': user_lr},
        {'params': model.item_embedding, 'lr': args.lr},
    ]

    optimizer = optim.Adam(param_groups)

    if args.coldstart or args.inductive:
        args.verbose = 1
        args.flag_step = 5
    for epoch in range(args.epoch):
        t1 = time()
        neg_g = construct_negative_graph(g, args.neg_samples, device=device)
        embedding_h = model(g)
            # print(embedding_h['item'].shape,embedding_h['item'][:data_generator.n_evaluate_items].shape)
            # sys.exit()

        bpr_loss, mf_loss, emb_loss = model.create_bpr_loss(pos_g, neg_g, embedding_h, loss_type='bpr')
        optimizer.zero_grad()
        bpr_loss.backward()
        optimizer.step()
        if (epoch + 1) % (args.verbose * 10) != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, bpr_loss, mf_loss, emb_loss, ssl_loss)
                print(perf_str)
            continue
        t2 = time()
        test_users = users_to_test
        if args.mix_train!=0:
            embedding_h['item'] = embedding_h['item'][:data_generator.n_evaluate_items]
        ret, _, _ = test_cpp(test_users, embedding_h)
        t3 = time()

        loss_loger.append(bpr_loss)
        rec_loger.append(ret['recall'])
        ndcg_loger.append(ret['ndcg'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, bpr_loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
        ret_recall_0 = ret['recall'][0]
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret_recall_0, cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=args.flag_step)

        # early stop
        if should_stop == True:
            break

    recs = np.array(rec_loger)
    # pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    # hit = np.array(hit_loger)
    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)
    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
    save_path = './output/%s_%s/%s.result' % (args.dataset, args.downstream, args.model_name)
    log(save_path, final_perf, args)

    if args.inductive:
        del g
        if args.hgcn == 1 and args.text_prompt==1:
            fc_w = model.fc.state_dict()
        elif args.lightgcn == 1:
            fc_w = model.w_pool.state_dict()
        else:
            fc_w = None
        coldstart.inductive(args, embedding_h, device, fc_w)
    elif args.coldstart:
        del g
        coldstart.coldstart(args, embedding_h['item'], device)


if __name__ == '__main__':
    seed = 1
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    args = parse_args()
    main(args)
