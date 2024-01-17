import random
import sys
import json
import numpy as np
import random as rd
import dgl
import torch
import pickle


class Data(object):
    def __init__(self, args, path, batch_size, dataset):
        downstream = args.downstream
        instructor = args.instructor
        inductive = args.inductive
        coldstart = args.coldstart
        self.path = path + '/split_induct_transduct/' if (inductive and dataset == 'amazon') else path
        self.batch_size = batch_size
        self.test_set = {}

        if dataset == 'goodreads':
            pretrain_type = 'comics'
            downstream_type = 'poetry'
        elif dataset == 'goodreads2':
            downstream_type = 'mystery'
            pretrain_type = 'pretrain'
        elif dataset == 'goodreads3':
            downstream_type = 'history'
            pretrain_type = downstream_type + '_pretrain'
        elif dataset == 'amazon':
            downstream_type = downstream
            pretrain_type = downstream_type + '_pretrain'
        if dataset in ['goodreads3', 'amazon']:
            item_split_file = path + '/' + downstream_type + '_item_split'
            with open(item_split_file, 'rb') as fp:
                self.item_split = pickle.load(fp)
        else:
            self.item_split = [42698]
        pretrain_path = 'user_%s_' % (pretrain_type)
        downstream_path = 'user_%s_' % (downstream_type)
        pretrain_file = self.path + '/' + pretrain_path + 'sampled.json'

        coldstart_train_file = self.path + '/coldstart_' + downstream_path + 'train.json'
        coldstart_test_file = self.path + '/coldstart_' + downstream_path + 'test.json'
        train_file = self.path + '/' + downstream_path + 'train.json'
        test_file = self.path + '/' + downstream_path + 'test.json'
        if instructor == 0:
            subfix = '_description_sampled.npy'
        elif instructor == 1:
            print('text embedding encoded by instructor_base')
            subfix = '_description_sampled_instructor.npy'
        elif instructor == 2:
            print('text embedding encoded by instructor_large')
            subfix = '_description_sampled_instructor_xl.npy'
        elif instructor == 3:
            print('text embedding encoded by gtr')
            subfix = '_description_sampled_gtr.npy'
        elif instructor == 4:
            print('text embedding encoded by bert')
            subfix = '_description_sampled_bert.npy'
        elif instructor == 5:
            print('text embedding encoded by e5')
            subfix = '_description_sampled_e5.npy'
        comics_text_file = path + '/%s' % (pretrain_type) + subfix
        poetry_text_file = path + '/%s' % (downstream_type) + subfix

        if args.classification == 1:
            cls_path = path + '/cls/'
            if dataset in ['amazon','goodreads3']:
                pretrain_file = cls_path + 'cls_' + pretrain_path + 'sampled.json'
                train_file = cls_path + 'cls_sampled_' + downstream_path + 'train.json'
                comics_text_file = cls_path + 'cls_pretrain_description.npy'
                poetry_text_file = cls_path + 'cls_downstream_description.npy'
                # print(comics_text_file, poetry_text_file)
                pretrain_split_file = cls_path + 'cls_pretrain_split'
                downstream_split_file = cls_path + 'cls_downstream_split'
                with open(pretrain_split_file, 'rb') as fp:
                    pretrain_split_tuples = pickle.load(fp)
                    self.pretrain_split = [i[1]-i[0] for i in pretrain_split_tuples]
                with open(downstream_split_file, 'rb') as fp:
                    downstream_split_tuples = pickle.load(fp)
                    self.downstream_split = [i[1]-i[0] for i in downstream_split_tuples]
                self.item_split = self.pretrain_split
            # sys.exit()

        if args.prompt_bytask:
            self.cate_prompt = torch.from_numpy(np.load(self.path + '/cate_prompt.npy'))
            self.downstream_cate_prompt = torch.from_numpy(np.load(self.path + '/downstream_cate_prompt.npy'))
        # print(self.downstream_cate_prompt)
        comics_text_embedding = torch.from_numpy(np.load(comics_text_file))
        # sys.exit()
        print('Pretrain item text data loaded.')
        item_text_embedding = torch.from_numpy(np.load(poetry_text_file))

        # print(comics_text_embedding.shape)
        # print(item_text_embedding.shape)
        # sys.exit()
        print('Item text data loaded.')
        # get number of users and items
        self.n_pre_users, self.n_comics = 0, 0
        self.n_cst_users = 0
        self.n_users, self.n_items = 0, 0

        self.n_all_users = 0

        self.n_pretrain, self.n_train, self.n_test = 0, 0, 0
        self.n_cst_train, self.n_cst_test = 0, 0
        self.cst_train_items, self.cst_test_set = {}, {}
        self.train_items, self.test_set = {}, {}
        user_comics_src, user_comics_dst = [], []
        cst_user_src, cst_user_dst = [], []
        user_item_src, user_item_dst = [], []
        with open(pretrain_file, 'r') as f:
            line = f.readline().strip()
            dict = json.loads(line)
            for k, v in dict.items():
                self.n_pre_users = max(self.n_pre_users, int(k))
                self.n_comics = max(self.n_comics, max(v))
                self.n_pretrain += len(v)
                user_comics_src.extend([int(k)] * len(v))
                user_comics_dst.extend(v)
        self.n_pre_users += 1
        self.n_comics += 1
        item_set = set([])
        with open(train_file, 'r') as f:
            line = f.readline().strip()
            dict = json.loads(line)
            for k, v in dict.items():
                self.n_users = max(self.n_users, int(k))
                item_set.update(v)
                self.n_items = max(self.n_items, max(v))
                self.n_train += len(v)
                self.train_items[int(k)] = v
                user_item_src.extend([int(k)] * len(v))
                user_item_dst.extend(v)
        if args.classification != 1:
            with open(test_file, 'r') as f:
                line = f.readline().strip()
                dict = json.loads(line)
                for k, v in dict.items():
                    self.n_users = max(self.n_users, int(k))
                    self.n_items = max(self.n_items, max(v))
                    item_set.update(v)
                    self.n_test += len(v)
                    self.test_set[int(k)] = v

        self.n_users += 1
        self.n_items += 1

        if args.classification == 1 and self.n_pre_users!= self.n_users:
            print('pretrained user number does not match finetuning user number!')
            self.n_pre_users = self.n_users

        self.print_pretrain_statistics()
        self.print_statistics()

        pretrain_data_dict = {
            ('user', 'ui', 'item'): (user_comics_src, user_comics_dst),
            ('item', 'iu', 'user'): (user_comics_dst, user_comics_src), }
        pretrain_num_dict = {
            'user': self.n_pre_users, 'item': self.n_comics
        }
        self.g_pretrain = dgl.heterograph(pretrain_data_dict, num_nodes_dict=pretrain_num_dict)
        self.g_pretrain.nodes['item'].data['tx'] = comics_text_embedding
        print('Pretrain text embedding tied to graph.')

        if args.mix_train != 0:
            sample_ratio = args.mix_train
            pretrain_list_idx = list(range(0, len(user_comics_dst)))
            sampled_edges_idx = rd.sample(pretrain_list_idx, round(self.n_pretrain * sample_ratio))
            # print(len(user_item_src))
            # print(len(sampled_edges_idx))
            user_item_src = user_item_src + [user_comics_src[i] for i in sampled_edges_idx]
            user_item_dst = user_item_dst + [i + self.n_items for i in [user_comics_dst[j] for j in sampled_edges_idx]]
            self.n_evaluate_items = self.n_items
            self.n_items += self.n_comics
            print('MIXED: num of mixed edges=%d, items after mixed=%d' % (len(user_item_src), self.n_items))
            # print(item_text_embedding.shape, comics_text_embedding.shape)
            item_text_embedding = torch.cat((item_text_embedding, comics_text_embedding), dim=0)
            # print(item_text_embedding.shape)
            # sys.exit()

        data_dict = {
            ('user', 'ui', 'item'): (user_item_src, user_item_dst),
            ('item', 'iu', 'user'): (user_item_dst, user_item_src), }
        num_dict = {
            'user': self.n_users, 'item': self.n_items
        }
        self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)
        self.g.nodes['item'].data['tx'] = item_text_embedding
        print('Text embedding tied to graph.')

        if inductive:
            self.n_all_users = self.n_users - 1
            with open(coldstart_train_file, 'r') as f:
                line = f.readline().strip()
                dict = json.loads(line)
                for k, v in dict.items():
                    if 0 < inductive < 1:
                        num_items_to_keep = round(len(v) * inductive)
                        v = random.sample(v, num_items_to_keep)
                    self.n_all_users = max(self.n_all_users, int(k))
                    self.n_cst_train += len(v)
                    self.cst_train_items[int(k)] = v
                    user_item_src.extend([int(k)] * len(v))
                    user_item_dst.extend(v)
            with open(coldstart_test_file, 'r') as f:
                line = f.readline().strip()
                dict = json.loads(line)
                for k, v in dict.items():
                    self.n_all_users = max(self.n_all_users, int(k))
                    self.n_cst_test += len(v)
                    self.cst_test_set[int(k)] = v
            self.n_all_users += 1
            self.print_induct_statistics()
            all_data_dict = {
                ('user', 'ui', 'item'): (user_item_src, user_item_dst),
                ('item', 'iu', 'user'): (user_item_dst, user_item_src), }
            all_n_num_dict = {
                'user': self.n_all_users, 'item': self.n_items
            }
            self.g_all = dgl.heterograph(all_data_dict, num_nodes_dict=all_n_num_dict)
            self.g_all.nodes['item'].data['tx'] = item_text_embedding
        elif coldstart:
            cst_set = set([])
            with open(coldstart_train_file, 'r') as f:
                line = f.readline().strip()
                dict = json.loads(line)
                for k, v in dict.items():
                    k = int(k) - self.n_users
                    cst_set.add(int(k))
                    self.n_cst_train += len(v)
                    self.cst_train_items[int(k)] = v
                    cst_user_src.extend([int(k)] * len(v))
                    cst_user_dst.extend(v)
            with open(coldstart_test_file, 'r') as f:
                line = f.readline().strip()
                dict = json.loads(line)
                for k, v in dict.items():
                    k = int(k) - self.n_users
                    cst_set.add(int(k))
                    self.n_cst_test += len(v)
                    self.cst_test_set[int(k)] = v
            self.n_cst_users = len(cst_set)

            self.print_cst_statistics()
            cst_data_dict = {
                ('user', 'ui', 'item'): (cst_user_src, cst_user_dst),
                ('item', 'iu', 'user'): (cst_user_dst, cst_user_src), }
            cst_n_num_dict = {
                'user': self.n_cst_users, 'item': self.n_items
            }
            self.g_cst = dgl.heterograph(cst_data_dict, num_nodes_dict=cst_n_num_dict)
            self.g_cst.nodes['item'].data['tx'] = item_text_embedding
            print('Cold-start user graph created.')

    def print_pretrain_statistics(self):
        print('n_pretrain_users=%d, n_pretrain_items=%d' % (
            self.n_pre_users, self.n_comics))
        print('n_pretrain=%d, sparsity=%.5f' % (
            self.n_pretrain, self.n_pretrain / (self.n_pre_users * self.n_comics)))

    def print_induct_statistics(self):
        print('n_all_users=%d, n_inductive_users=%d,' % (
            self.n_all_users, self.n_all_users - self.n_users))
        print('n_inductive_train=%d, n_inductive_test=%d,sparsity=%.5f' % (
            self.n_cst_train, self.n_cst_test,
            (self.n_cst_train + self.n_cst_test) / ((self.n_all_users - self.n_users) * self.n_items)))

    def print_cst_statistics(self):
        print('n_coldstart_users=%d' % (
            self.n_cst_users))
        print('n_coldstart_train=%d, n_coldstart_test=%d,sparsity=%.5f' % (
            self.n_cst_train, self.n_cst_test,
            (self.n_cst_train + self.n_cst_test) / (self.n_cst_users * self.n_items)))

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (
            self.n_users, self.n_items))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u] + self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
