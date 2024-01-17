import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--data_path', nargs='?', default='./Data/',
                        help='Input data path.')
    parser.add_argument('--weights_path', nargs='?', default='./Weights/',
                        help='Input data path.')
    parser.add_argument('--model_name', type=str, default='LightGCN',
                        help='Saved model name.')
    parser.add_argument('--dataset', nargs='?', default='amazon',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    parser.add_argument('--downstream', nargs='?', default='musical')
    parser.add_argument('--classification', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation. 1 for all')
    parser.add_argument('--epoch', type=int, default=10000,
                        help='Number of epoch.')
    parser.add_argument('--mix_train',type=float, default=0)
    parser.add_argument('--user_split',type=int, default=0)
    parser.add_argument('--whitening',type=int, default=0)
    parser.add_argument('--prompt',type=int,default=0)
    parser.add_argument('--mf_au', type=int, default=0)
    parser.add_argument('--pre_gcn',type=int,default=0)
    parser.add_argument('--fuse_decay',type=float,default=0.1)
    parser.add_argument('--fuse_decay_user',type=float,default=0.0001)
    parser.add_argument('--show_distance',type=int, default=0)
    parser.add_argument('--loss', type=str, default='bpr')

    '''pretrain framework baselines, NEED to SET user_pretrain = 1'''
    parser.add_argument('--gcc', type=int, default=0)
    parser.add_argument('--graphmae', type=int, default=0)
    parser.add_argument('--attrimask', type=int, default=0)


    '''Gnn&HGnn framework baselines, NEED to SET user_pretrain = 0'''
    parser.add_argument('--lightgcn', type=int, default=0,
                        help='1:LGCN, 2:SGL, 3:DirectAU, 1&inductive:GraphSage-max')
    parser.add_argument('--gat',type=int, default=0) # Graphformer,  Need to SET item_text=1
    parser.add_argument('--hgcn', type=int, default=1,
                        help='1:HGNN, 2:HCCF, 3:DHCF')
    parser.add_argument('--ssl', type=int, default=0)
    parser.add_argument('--pretrain_epoch_num', type=int, default=300,
                        help='Number of epoch.Amazon:300, Goodread(3): 200')

    parser.add_argument('--coldstart',type=int, default=0)
    parser.add_argument('--inductive',type=float, default=0)

    parser.add_argument('--item_text',type=int,default=1)
    parser.add_argument('--instructor',type=int,default=0)
    parser.add_argument('--prompt_bytask',type=int,default=0)

    parser.add_argument('--edge_prompt',type=int,default=0)
    parser.add_argument('--multitask_train',type=int, default=0,
                        help='only valid if pre_train = 1')
    parser.add_argument('--text_prompt', type=int, default=1,
                        )
    parser.add_argument('--user_pretrain', type=int, default=1,
                        help='0 for not using pretrained user embeddings, 1 for finetuning pretrained user embeddings,2 for frozen pretrained user embeddings')

    parser.add_argument('--norm_2', type=int, default=-1,
                        help='-0.5 for mx, -1 for cn/mx_C2EP')

    parser.add_argument('--pre_lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--user_lr', type=float, default=0.1,
                        help='user Learning rate. equal to lr if set to -1')
    parser.add_argument('--layer_num', type=int, default=3,
                        help='Output sizes of every layer')
    parser.add_argument('--regs', nargs='?', default='[1e-7]',
                        help='Regularizations.')

    parser.add_argument('--hgcn_u_hyperedge', type=int, default=1,
                        help='Hypergraph conv on user-group side with user as hyperedge and group as vertex')
    parser.add_argument('--user_hpedge_ig', type=int, default=0,
                        help='Hypergraph conv on user side with user as vertex'
                             '{0: only user-item conv with item as hyperedge;'
                             ' 1: simultaneous user-item and user-group conv;'
                             ' 2: sequential user-item and user-group conv};'
                            '3&4:HGNN and HGNN+')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')

    parser.add_argument('--batch_size', type=int, default=32768,
                        help='Batch size.')


    parser.add_argument('--neg_samples', type=int, default=1,
                        help='Number of negative samples.')

    parser.add_argument('--flag_step', type=int, default=10,
                        help='Number of negative samples.amazon:10, goodreads3:5, goodreads:10')

    parser.add_argument('--gpu', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--Ks', nargs='?', default='[10,20]',
                        help='Output sizes of every layer')

    parser.add_argument('--fast_test', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='full',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    return parser.parse_args()
