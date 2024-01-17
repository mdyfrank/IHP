import os
import re
import sys

import numpy as np



def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def log(save_path, final_perf, args):
    ensureDir(save_path)
    f = open(save_path, 'a')
    f.write(
        'top_k=%s, lr=%.4f, pre_lr=%.4f, user_lr=%.6f, layer_num=%s, batch_size=%d, norm=%.1f,  pretrain_epoch_num=%d, loss=%s, lightgcn=%d, gat=%d, '
        'num_neg=%d, hgcn=%d,  text_prompt=%d, user_pretrain=%d, item_text=%d, instructor=%d, prompt_bytask=%d '
        'verbose=%d, fuse_decay=%.5f, fuse_decay_user=%.5f, inductive=%.2f, mix_train=%.2f, regs=%s\n\t%s\n'
        % (args.Ks, args.lr, args.pre_lr, args.user_lr, args.layer_num, args.batch_size, args.norm_2, args.pretrain_epoch_num,
           args.loss,
           args.lightgcn,
           args.gat,
           args.neg_samples,
           args.hgcn,
           args.text_prompt,
           args.user_pretrain,
           args.item_text,
           args.instructor,
           args.prompt_bytask,
           args.verbose, args.fuse_decay, args.fuse_decay_user,
           args.inductive,
           args.mix_train,
           args.regs, final_perf))
    f.close()

def random_batch_users(users_to_test, batch_size):
    user_batch = np.random.choice(users_to_test,batch_size)
    return user_batch

def convert_dict_str(dict):
    s = '|'
    for key, value in dict.items:
        s += 'group %d: %f|'%(key, value[0])
    return s

def convert_dict_list(dict):
    s = []
    for i in sorted(dict.keys()):
        s.append(dict[i][0])
    return s

def convert_list_str(list):
    s = ''
    for i in list:
        s += '|%.5f|'%(i)
    return s

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop