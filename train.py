import time
import argparse
import numpy as np
import torch
#from deeprobust.graph.defense import GCN, ProGNN
from model.gcn import GCN
from prognn.prognn import ProGNN
#from deeprobust.graph.data import Dataset, PrePtbDataset
#from deeprobust.graph.utils import preprocess, encode_onehot, get_train_val_test

from dataset_process.dataset import Dataset
from dataset_process.attacked_data import PrePtbDataset
from dataset_process.utils import preprocess, encode_onehot, get_train_val_test
import os.path

from graph_cert.utils import *
from dataset_process.get_fragile import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=574, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora_Attackinduce',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='no',
        choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=20, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=0.000, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=0, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=2, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')
parser.add_argument('--p_robust', type=float, default=0.9, help='the robust threshold for the co-learning model')
parser.add_argument('--local_budget', type=int, default=1, help='the local budget')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print (args.cuda)
device = torch.device("cuda" if args.cuda else "cpu")
#device = torch.device("cpu")

alpha = 0.85

if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

print(args)

np.random.seed(15) # Here the random seed is to split the train/val/test data, we need to set the random seed to be the same as that when you generate the perturbed graph

my_path = os.path.abspath(os.path.dirname(__file__))
if args.dataset.find("cora")!=-1:
    root = os.path.join(my_path,'dataset/cora/')
if args.dataset.find("citeseer")!=-1:
    root = os.path.join(my_path, 'dataset/citeseer/')
if args.dataset.find("dbpedia")!=-1:
    root = os.path.join(my_path, 'dataset/dbpedia/')
if args.dataset.find("yelp")!=-1:
    root = os.path.join(my_path, 'dataset/yelp/')
if args.dataset.find('oag')!=-1:
    root = os.path.join(my_path, 'dataset/oag/')
data = Dataset(root, name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

if args.dataset == 'pubmed':

    idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0],
            val_size=0.1, test_size=0.8, stratify=encode_onehot(labels), seed=15)

if args.attack == 'no':
    perturbed_adj = adj
    ppr = torch.FloatTensor(propagation_matrix(perturbed_adj, alpha=alpha)).to(device)

if args.attack == 'random':
    from deeprobust.graph.global_attack import Random
    attacker = Random()
    n_perturbations = int(args.ptb_rate * (adj.sum()//2))
    perturbed_adj = attacker.attack(adj, n_perturbations, type='add')

if args.attack == 'meta' or args.attack == 'nettack':
    perturbed_data = PrePtbDataset(root='/tmp/',
            name=args.dataset,
            attack_method=args.attack,
            ptb_rate=args.ptb_rate)
    perturbed_adj = perturbed_data.adj
    if args.attack == 'nettack':
        idx_test = perturbed_data.target_nodes

np.random.seed(args.seed)
torch.manual_seed(args.seed)

model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout, device=device)

#adver_config=None

# set the local budget proportional to the degree
deg = adj.sum(1).A1.astype(np.int32)
local_strength = 9
local_budget = np.maximum(deg - 11 + local_strength, 0)

'''
####Add only to test the impact of local budget
local_budget = np.ones_like(local_budget)
local_budget = local_budget *args.local_budget
'''



global_budget =  5000

#choose the fast heuristic version of fragile:                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ragile = get_fast_heu_fragile(root,args.dataset)
#fragile = get_fragile(adj=adj, threat_model='rem')
fragile = get_new_fragile(root,args.dataset,adj,global_budget)
print("fragile shape")
print(fragile.shape)


adver_config = {
    'alpha': alpha,
    'adj_matrix': perturbed_adj,
    'fragile': fragile,
    'local_budget': local_budget,
    'margin': 1
}

#adver_config = None

adver_config['loss_type'] = 'cem'


if args.only_gcn:
    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, sparse=True, device=device)
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, verbose=True,  rain_iters=args.epochs)
    model.test(idx_test)
elif adver_config is None:
    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)
    #unit test for updated GCN
    #test_1, logits_test, diffused_logits_test = model(features, perturbed_adj, ppr)
    #print(test_1.shape)
    prognn = ProGNN(model, args, device)
    prognn.fit(features, perturbed_adj, labels, idx_train, idx_val,ppr)
    prognn.test(features, labels, idx_test,ppr)
    
else:
    print("start testing")
    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)
    prognn = ProGNN(model, args, device)
    prognn.new_fit(features, perturbed_adj, labels, idx_train, idx_val, ppr,adver_config)
    prognn.test(features, labels, idx_test, ppr)




