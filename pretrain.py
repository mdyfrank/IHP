import torch.optim as optim
from utility.batch_test import *
from utility.helper import early_stopping, random_batch_users, ensureDir, convert_dict_list, convert_list_str
from model import *
from time import time

def pretrain(args, device):
    g = data_generator.g_pretrain
    g = g.to(device)
    pos_g = construct_user_item_bigraph(g)
    # print(g.nodes['item'].data['tx'])
    if args.prompt_bytask:
        # model = PromptGCN(args, g, device, item_split=data_generator.item_split).to(device)
        model = PromptGCN(args, g, device, item_split=data_generator.item_split, pretrain_cate_prompt=data_generator.cate_prompt, pretrain=True).to(device)
        # model = PromptGCN(args, g, device, item_split=data_generator.item_split, pretrain=True).to(device)
    else:
        model = PromptGCN(args, g, device, pretrain=True).to(device)
    print('pretrain.py')
    pre_train_best, stopping_step = float('inf'), 0
    optimizer = optim.Adam(model.parameters(), lr=args.pre_lr)
    # flag_step = args.flag_step
    ssl_loss = 0
    if args.pretrain_epoch_num!=0:
        num_epoch = args.pretrain_epoch_num
    else:
        num_epoch = args.epoch
    t0 = time()
    for epoch in range(num_epoch):
        # args.flag_step = 5
        t1 = time()
        neg_g = construct_negative_graph(g, args.neg_samples, device=device)
        embedding_h = model(g)
        bpr_loss, mf_loss, emb_loss = model.create_bpr_loss(pos_g, neg_g, embedding_h, loss_type='bpr')
        # print(bpr_loss)
        if args.ssl:
            # ssl_loss = 1e-7 * (model.create_ssl_loss_user(ssl_temp = 0.1) + model.create_ssl_loss_batched_item(ssl_temp = 0.1,k=16))
            ssl_loss = model.create_ssl_loss_user(ssl_temp = 0.05)
            # ssl_loss = 1e-7 * (model.create_ssl_loss_user(ssl_temp = 0.1) + model.create_ssl_loss_item(ssl_temp = 0.1))
            bpr_loss += ssl_loss
        optimizer.zero_grad()
        bpr_loss.backward()
        optimizer.step()
        # if (epoch + 1) % (args.verbose * 10) != 0:
        #     if args.verbose > 0 and epoch % args.verbose == 0:
        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
            epoch, time() - t1, bpr_loss, mf_loss, emb_loss, ssl_loss)
        print(perf_str)
            # continue
        pre_train_best, stopping_step, should_stop = early_stopping(bpr_loss, pre_train_best,
                                                                    stopping_step, expected_order='dec',
                                                                    flag_step=args.flag_step)
        if should_stop == True:
            print('Pre-train stopped.')
            break
    t2 = time()
    print('Pretraining time:',t2-t0)
    # torch.save(model.fc.state_dict(),'linear_layer_weights.pth')
    fc_w = model.fc.state_dict()
    edge_w = model.edge_w if args.edge_prompt else None
    pretrain_user_embedding = embedding_h['user']
    whitening_w = None
    if args.whitening == 1:
        whitening_w = [model.whitening_w1.state_dict(),model.whitening_w2.state_dict(),model.whitening_w3.state_dict(),model.whitening_bias]
    del g, pos_g, neg_g, model, optimizer, embedding_h
    torch.cuda.empty_cache()
    return fc_w,pretrain_user_embedding, edge_w, whitening_w