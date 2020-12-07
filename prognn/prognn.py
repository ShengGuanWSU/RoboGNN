import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from deeprobust.graph.utils import accuracy
from dataset_process.utils import accuracy, cal_scores
#from deeprobust.graph.defense.pgd import PGD, prox_operators
from pgd.pgd import PGD, prox_operators
import warnings
from graph_cert.certify import k_squared_parallel
from graph_cert.certify_training import *
from scipy import sparse as sp


class ProGNN:
    """ ProGNN (Properties Graph Neural Network). See more details in Graph Structure Learning for Robust Graph Neural Networks, KDD 2020, https://arxiv.org/abs/2005.10203.

    Parameters
    ----------
    model:
        model: The backbone GNN model in ProGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
    See details in https://github.com/ChandlerBang/Pro-GNN.

    """

    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)
        self.p_robust = args.p_robust
        self.compt_robust = 0.0
    '''
    def fit(self, features, adj, labels, idx_train, idx_val, ppr,**kwargs):
        """Train Pro-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
        args = self.args

        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        estimator = EstimateAdj(adj, symmetric=args.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=args.lr_adj)

        self.optimizer_l1 = PGD(estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=args.lr_adj, alphas=[args.alpha])

        # warnings.warn("If you find the nuclear proximal operator runs too slow on Pubmed, you can  uncomment line 67-71 and use prox_nuclear_cuda to perform the proximal on gpu.")
        # if args.dataset == "pubmed":
        #     self.optimizer_nuclear = PGD(estimator.parameters(),
        #               proxs=[prox_operators.prox_nuclear_cuda],
        #               lr=args.lr_adj, alphas=[args.beta])
        # else:
        warnings.warn("If you find the nuclear proximal operator runs too slow, you can modify line 77 to use prox_operators.prox_nuclear_cuda instead of prox_operators.prox_nuclear to perform the proximal on GPU. See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.optimizer_nuclear = PGD(estimator.parameters(),
                    proxs=[prox_operators.prox_nuclear_cuda],
                    lr=args.lr_adj, alphas=[args.beta])
        else:
            self.optimizer_nuclear = PGD(estimator.parameters(),
                    proxs=[prox_operators.prox_nuclear],
                    lr=args.lr_adj, alphas=[args.beta])
            

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            if args.only_gcn:
                self.train_gcn(epoch, features, estimator.estimated_adj,
                        labels, idx_train, idx_val)
            else:
                for i in range(int(args.outer_steps)):
                    self.train_adj(epoch, features, adj, labels,
                            idx_train, idx_val,ppr)

                for i in range(int(args.inner_steps)):
                    self.train_gcn(epoch, features, estimator.estimated_adj,
                            labels, idx_train, idx_val,ppr)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
    '''

    def new_fit(self, features, adj, labels, idx_train, idx_val, ppr, adver_config, **kwargs):
        """Train Pro-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        ppr :
            pageRank vectors
        adver_config :
            adver_config file
        """
        args = self.args

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.lr, weight_decay=args.weight_decay)
        estimator = EstimateAdj(adj, symmetric=args.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                                       momentum=0.9, lr=args.lr_adj)

        '''
        self.optimizer_l1 = PGD(estimator.parameters(),
                                proxs=[prox_operators.prox_l1],
                                lr=args.lr_adj, alphas=[args.alpha])
        '''

        # warnings.warn("If you find the nuclear proximal operator runs too slow on Pubmed, you can  uncomment line 67-71 and use prox_nuclear_cuda to perform the proximal on gpu.")
        # if args.dataset == "pubmed":
        #     self.optimizer_nuclear = PGD(estimator.parameters(),
        #               proxs=[prox_operators.prox_nuclear_cuda],
        #               lr=args.lr_adj, alphas=[args.beta])
        # else:
        '''
        warnings.warn(
            "If you find the nuclear proximal operator runs too slow, you can modify line 77 to use prox_operators.prox_nuclear_cuda instead of prox_operators.prox_nuclear to perform the proximal on GPU. See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        if torch.cuda.is_available():
            self.optimizer_nuclear = PGD(estimator.parameters(),
                    proxs=[prox_operators.prox_nuclear_cuda],
                    lr=args.lr_adj, alphas=[args.beta])
        else:
            self.optimizer_nuclear = PGD(estimator.parameters(),
                    proxs=[prox_operators.prox_nuclear],
                    lr=args.lr_adj, alphas=[args.beta])
        
        '''

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            if args.only_gcn:
                self.train_gcn(epoch, features, estimator.estimated_adj,
                               labels, idx_train, idx_val)
            else:
                for i in range(int(args.outer_steps)):
                    '''
                    self.train_adj(epoch, features, adj, labels,
                                   idx_train, idx_val, ppr)
                    '''
                    self.new_train_adj(epoch, features, adj,labels,
                                       idx_train, idx_val, ppr, adver_config)
                    if self.compt_robust >=self.p_robust:
                        print(''''early stopping since robust criteria is fulfilled''')
                        break

                for i in range(int(args.inner_steps)):
                    '''
                    self.train_gcn(epoch, features, estimator.estimated_adj,
                                   labels, idx_train, idx_val, ppr)
                    '''
                    self.new_train_gcn(epoch, features, estimator.estimated_adj,
                                       labels, idx_train, idx_val, ppr,adver_config)
                    if self.compt_robust >= self.p_robust:
                        print(''''early stopping since robust criteria is fulfilled''')
                        break
            if self.compt_robust >= self.p_robust:
                break       

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
    '''
    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val,ppr):
        args = self.args
        estimator = self.estimator
        adj = estimator.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        #output = self.model(features, adj)

        print(features.device)
        print(adj.device)
        print(ppr.device)
        output, logits, diffused_logits = self.model(features, adj, ppr)
        #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        #acc_train = accuracy(output[idx_train], labels[idx_train])

        print(diffused_logits.device)
        print(labels.device)
        loss_train = F.cross_entropy(diffused_logits[idx_train], labels[idx_train])
        acc_train = accuracy(diffused_logits[idx_train], labels[idx_train])


        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()

        #output = self.model(features, adj)
        output, logits, diffused_logits = self.model(features, adj, ppr)

        #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        #acc_val = accuracy(output[idx_val], labels[idx_val])

        loss_val = F.cross_entropy(diffused_logits[idx_val], labels[idx_val])
        acc_val = accuracy(diffused_logits[idx_val], labels[idx_val])


        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))
    '''
    def new_train_gcn(self, epoch, features, adj, labels, idx_train, idx_val,ppr,adver_config):
        args = self.args
        estimator = self.estimator
        adj = estimator.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        ##combine the idx_train and idx_val
        idx_observed = np.concatenate((idx_train, idx_val))
        arng = torch.arange(len(idx_observed))
        reference_labels = labels[idx_observed]

        #print(features.device)
        #print(adj.device)
        #print(ppr.device)


        #output = self.model(features, adj)

        output, logits, diffused_logits = self.model(features, adj, ppr)
        #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        #acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train = F.cross_entropy(diffused_logits[idx_train], labels[idx_train])
        

        # add the certificates
        #retrieved_adj = estimator.estimated_adj.data.cpu().detach().numpy()
        #retrieved_adj = sp.csr_matrix(retrieved_adj)

        '''
        if adver_config is not None:
            # change the adver_config['adj_matrix'] to normalized_adj

            
            k_squared_pageranks = k_squared_parallel(adj=retrieved_adj, alpha=adver_config['alpha'],
                                                     fragile=adver_config['fragile'],
                                                     local_budget=adver_config['local_budget'],
                                                     logits=logits.cpu().detach().numpy(),
                                                     nodes=idx_observed)
            
            k_squared_pageranks = k_squared_parallel(adj=adver_config['adj_matrix'], alpha=adver_config['alpha'],
                                                     fragile=adver_config['fragile'],
                                                     local_budget=adver_config['local_budget'],
                                                     logits=logits.cpu().detach().numpy(),
                                                     nodes=idx_observed)
            
            
            # worst-case/adversarial logits
            adv_logits = get_adv_logits(logits=logits, reference_labels=reference_labels, arng=arng,
                                        k_squared_pageranks=k_squared_pageranks)
        '''
        #print(diffused_logits.device)
        #print(labels.device)
        #print(adv_logits.device)
        '''
        for parameter in self.model.parameters():
            print(parameter)
        '''
        '''
        if adver_config['loss_type'] == 'cem':
            margin = adver_config['margin']
            loss_train = cem_loss(diffused_logits=diffused_logits[idx_train], adv_logits=adv_logits[:len(idx_train)],
                                labels=labels[idx_train], margin=margin).float()
    
            #loss_train = F.cross_entropy(diffused_logits[idx_train], labels[idx_train])
            #print("loss train device")
            #print(loss_train.device)
            #print(loss_train.grad)
            #print(loss_train.grad_fn)
        else:
            raise ValueError('loss type not recognized.')
        
        # compute the fraction of train/val nodes which are certifiably robust
        worst_margins = adv_logits.clone()
        worst_margins[arng, reference_labels] = float('inf')
        p_robust = (worst_margins.min(1).values > 0).sum().item() / len(reference_labels)
        ##update the computation robust
        if p_robust > self.compt_robust:
            self.compt_robust = p_robust
        '''


        acc_train = accuracy(diffused_logits[idx_train], labels[idx_train])

        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()

        #output = self.model(features, adj)
        output, logits, diffused_logits = self.model(features, adj, ppr)

        #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        #acc_val = accuracy(output[idx_val], labels[idx_val])
        

        loss_val = F.cross_entropy(diffused_logits[idx_val], labels[idx_val])
        
        '''
        loss_val = cem_loss(diffused_logits=diffused_logits[idx_val], adv_logits=adv_logits[len(idx_train):],
                            labels=labels[idx_val], margin=margin)
        '''

        acc_val = accuracy(diffused_logits[idx_val], labels[idx_val])


        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            ##update the computation robust
            #self.compt_robust = p_robust
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val.double() < self.best_val_loss:
            self.best_val_loss = loss_val.double()
            '''
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            '''
            if args.debug:
                print(f'\t=== NOT saving current graph/gcn, but val_loss improved, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      #'robust: {:.8f}'.format(p_robust),
                      'time: {:.4f}s'.format(time.time() - t))


    '''
    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val,ppr):
        estimator = self.estimator
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        #because we have this step, no need to precompute the normalized_adj first
        normalized_adj = estimator.normalize()

        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        output,  logits, diffused_logits= self.model(features, normalized_adj,ppr)
        #loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])

        loss_gcn = F.cross_entropy(diffused_logits[idx_train], labels[idx_train])


        #acc_train = accuracy(output[idx_train], labels[idx_train])

        acc_train = accuracy(diffused_logits[idx_train], labels[idx_train])

        loss_symmetric = torch.norm(estimator.estimated_adj \
                        - estimator.estimated_adj.t(), p="fro")

        loss_diffiential =  loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat + args.phi * loss_symmetric

        loss_diffiential.backward()

        self.optimizer_adj.step()
        loss_nuclear =  0 * loss_fro
        if args.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                    + args.gamma * loss_gcn \
                    + args.alpha * loss_l1 \
                    + args.beta * loss_nuclear \
                    + args.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        normalized_adj = estimator.normalize()
        #output = self.model(features, normalized_adj)

        output, logits, diffused_logits = self.model(features, normalized_adj, ppr)


        #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        loss_val = F.cross_entropy(diffused_logits[idx_val], labels[idx_val])

        #acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_val = accuracy(diffused_logits[idx_val], labels[idx_val])

        print('Epoch: {:04d}'.format(epoch+1),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())
        
        
        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj-adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))
    '''
    def new_train_adj(self, epoch, features, adj, labels, idx_train, idx_val,ppr,adver_config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        estimator = self.estimator
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        #estimitor.train() sets the mode to train. So effectively layers like dropout, batchnorm etc. which behave different on the train and test procedures know what is going on and hence can behave accordingly
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        #because we have this step, no need to precompute the normalized_adj first
        normalized_adj = estimator.normalize()

        ##combine the idx_train and idx_val
        idx_observed = np.concatenate((idx_train, idx_val))
        arng = torch.arange(len(idx_observed))
        reference_labels = labels[idx_observed]
        #trace_val=[]


        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        output,  logits, diffused_logits= self.model(features, normalized_adj,ppr)

        #add the certificates
        if adver_config is not None:
            #change the adver_config['adj_matrix'] to normalized_adj

            #retrieved_adj=estimator.estimated_adj.cpu().detach().numpy()
            #retrieved_adj = sp.csr_matrix(retrieved_adj)
            
            k_squared_pageranks = k_squared_parallel(adj=adver_config['adj_matrix'], alpha=adver_config['alpha'],
                                                     fragile=adver_config['fragile'],
                                                     local_budget=adver_config['local_budget'],
                                                     logits=logits.cpu().detach().numpy(),
                                                     nodes=idx_observed)
            '''
            k_squared_pageranks = k_squared_parallel(adj=retrieved_adj, alpha=adver_config['alpha'],
                                                     fragile=adver_config['fragile'],
                                                     local_budget=adver_config['local_budget'],
                                                     logits=logits.cpu().detach().numpy(),
                                                     nodes=idx_observed)
            '''
            # worst-case/adversarial logits
            adv_logits = get_adv_logits(logits=logits, reference_labels=reference_labels, arng=arng,
                                        k_squared_pageranks=k_squared_pageranks)

        if adver_config['loss_type'] == 'cem':
            margin = adver_config['margin']
            loss_gcn = cem_loss(diffused_logits=diffused_logits[idx_train], adv_logits=adv_logits[:len(idx_train)],
                                  labels=labels[idx_train], margin=margin).float()

        else:
            raise ValueError('loss type not recognized.')

        # compute the fraction of train/val nodes which are certifiably robust
        worst_margins = adv_logits.clone()
        worst_margins[arng, reference_labels] = float('inf')
        p_robust = (worst_margins.min(1).values > 0).sum().item() / len(reference_labels)
        ##update the computation robust
        if p_robust > self.compt_robust:
            self.compt_robust = p_robust



        '''
        #loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])

        loss_gcn = F.cross_entropy(diffused_logits[idx_train], labels[idx_train])
        '''


        #acc_train = accuracy(output[idx_train], labels[idx_train])

        acc_train = accuracy(diffused_logits[idx_train], labels[idx_train])

        loss_symmetric = torch.norm(estimator.estimated_adj \
                        - estimator.estimated_adj.t(), p="fro")

        loss_diffiential =  loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat + args.phi * loss_symmetric
        #print("Pass loss train ")
        #print(loss_diffiential.device)
        #print(loss_diffiential.grad)
        #print(loss_diffiential.grad_fn)

        loss_diffiential.backward()

        self.optimizer_adj.step()
        loss_nuclear =  0 * loss_fro
        if args.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        '''
        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()
        '''

        total_loss = loss_fro \
                    + args.gamma * loss_gcn \
                    + args.alpha * loss_l1 \
                    + args.beta * loss_nuclear \
                    + args.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        normalized_adj = estimator.normalize()
        #output = self.model(features, normalized_adj)

        output, logits, diffused_logits = self.model(features, normalized_adj, ppr)

        '''

        #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        loss_val = F.cross_entropy(diffused_logits[idx_val], labels[idx_val])
        '''


        loss_val = cem_loss(diffused_logits=diffused_logits[idx_val], adv_logits=adv_logits[len(idx_train):],
                            labels=labels[idx_val], margin=margin)



        #acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_val = accuracy(diffused_logits[idx_val], labels[idx_val])

        print('Epoch: {:04d}'.format(epoch+1),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'robust: {:.8f}'.format(p_robust),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            ##update the computation robust
            self.compt_robust = p_robust
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val.double() < self.best_val_loss:
            self.best_val_loss = loss_val.double()
            '''
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            '''
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())
        #print(estimator.estimated_adj.device)
        #print(adj.device)
        
        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj.to(device)-adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))


    def test(self, features, labels, idx_test,ppr):
        """Evaluate the performance of ProGNN on test set
        """
        print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        if self.best_graph is None:
            adj = self.estimator.normalize()
        #output = self.model(features, adj)

        output, logits, diffused_logits = self.model(features, adj, ppr)

        #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        #acc_test = accuracy(output[idx_test], labels[idx_test])

        loss_test = F.cross_entropy(diffused_logits[idx_test], labels[idx_test])
        acc_test = accuracy(diffused_logits[idx_test], labels[idx_test])

        precision, recall, fscore, support, balanced_accuracy_score = cal_scores(diffused_logits[idx_test], labels[idx_test])

        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()),
              "precision = {:.4f}".format(precision.item()),
              "recall = {:.4f}".format(recall.item()),
              "fscore = {:.4f}".format(fscore.item()),
              "balanced_accuracy = {:.4f}".format(balanced_accuracy_score.item()))

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj.to(self.device)

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

