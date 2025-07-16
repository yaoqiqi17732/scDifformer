import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import time
import multiprocessing
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import copy
#from audtorch.metrics.functional import pearsonr as pcc_torch
from WB_gcn import *

class conGraphConvolutionlayer(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(conGraphConvolutionlayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support) #sparse matrix multiple
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphSAGE(nn.Module):
    def __init__(self,nfeat,
                 # 加入 embedding
                 nfeat_embed,
                 nhid,
                 nhid_embed,
                 gcn_hid_layers_num,
                 fcnn_hid_layers_num,
                 dropout,
                 nout1,
                 training,
                 policy,
                 gcn,
                 use_embedfeature
                ):
        super(GraphSAGE, self).__init__()
        self.nfeat = nfeat
        # embedding
        self.nfeat_embed = nfeat_embed
        self.nhid = nhid
        self.nhid_embed = nhid_embed
        self.gcn_hid_layers_num = gcn_hid_layers_num
        self.fcnn_hid_layers_num = fcnn_hid_layers_num
        self.dropout = dropout
        self.nout1 = nout1
        self.training = training
        self.policy = policy
        self.gcn = gcn
        self.use_embedfeature = use_embedfeature

        self.gc_in_exp = nn.Linear(self.nfeat, self.nhid, bias=True) #FCL
        self.bn_node_in_exp = nn.BatchNorm1d(nhid) # batchnorm
        # embedding
        if self.use_embedfeature == True:
            self.gc_in_embed = nn.Linear(self.nfeat_embed, self.nhid_embed)
            self.bn_node_in_embed = nn.BatchNorm1d(nhid_embed)
        
        # GraphSAGE
        self.graphsage_layer = nn.ModuleList([gs_block(self.nhid,self.nhid,self.policy,self.gcn) for i in range(self.gcn_hid_layers_num)])
        self.jknet=nn.Sequential(nn.LSTM(self.nhid,self.nhid,2),SelectItem(0))
        
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                # embedding
                if self.use_embedfeature == True:
                    exec('self.gc_out11{} = nn.Linear(self.nhid+self.nhid_embed, self.nhid+self.nhid_embed, bias=True)'.format(i+1))
                    exec('self.bn_out11{} = nn.BatchNorm1d(self.nhid+self.nhid_embed)'.format(i+1))
                else:
                    exec('self.gc_out11{} = nn.Linear(self.nhid, self.nhid, bias=True)'.format(i+1))
                    exec('self.bn_out11{} = nn.BatchNorm1d(self.nhid)'.format(i+1))
        # embedding
        if self.use_embedfeature == True:
            self.gc_out12 = nn.Linear(self.nhid+self.nhid_embed, self.nout1, bias=True)
        else:
            self.gc_out12 = nn.Linear(self.nhid, self.nout1, bias=True)
        # embedding

    
    def forward(self, x, embed, adj):  
        self.x = x
        self.x = self.gc_in_exp(self.x)
        self.x = self.bn_node_in_exp(self.x)
        self.x = F.elu(self.x)
        self.x = F.dropout(self.x, self.dropout, training = self.training)
        
        jk = []
        for layer in self.graphsage_layer:
            self.x = layer(self.x, adj)
            jk.append(self.x.unsqueeze(0))
        self.x = torch.cat(jk, 0)
        self.x = self.jknet(self.x).mean(0)
        self.x = self.bn_node_in_exp(self.x)
        self.x = F.elu(self.x)
        self.x = F.dropout(self.x, self.dropout, training=self.training)

        # 将embedding加入
        if self.use_embedfeature == True:
            #在 加入前加一个全连接
            self.embed = embed
            self.embed = self.gc_in_embed(self.embed)
            self.embed = self.bn_node_in_embed(self.embed)
            self.embed = F.elu(self.embed)
            self.embed = F.dropout(self.embed, self.dropout, training = self.training)

            self.x = torch.cat([self.x, self.embed], dim=1)

        
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec("self.x = self.gc_out11{}(self.x)".format(i+1))
                exec("self.x = self.bn_out11{}(self.x)".format(i+1))
                self.x = F.elu(self.x)
                self.x = F.dropout(self.x, self.dropout, training=self.training)
        
        self.x = self.gc_out12(self.x)

        return F.log_softmax(self.x, dim=1)

def GraphSAGE_train(model, 
                 train_valid_len,
                 test_len, 
                 feature,
                 embed_feature,
                 adj, 
                 label, 
                 epoch_n, 
                 loss_fn, 
                 optimizer, 
                 train_valid_ratio = 0.9,
                 scheduler = None,
                 early_stopping_patience = 5,
                 clip_grad_max_norm = 1,
                 load_test_groundtruth = False,
                 print_epoch_step = 1,
                 cpu_num = -1,
                 GCN_device = 'CPU'
                ):
    
    if GCN_device == 'CPU':
        device = torch.device("cpu")
        print('Use CPU as device.')
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('Use GPU as device.')
        else:
            device = torch.device("cpu")
            print('Use CPU as device.')
    
    if cpu_num == -1:
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
    else:
        torch.set_num_threads(cpu_num)
    
    model = model.to(device)
    adj = adj.to(device)
    feature = feature.to(device)
    # 加入 embedding
    embed_feature = embed_feature.to(device)
    label = label.to(device)
    
    time_open = time.time()

    train_idx = range(int(train_valid_len*train_valid_ratio))
    valid_idx = range(len(train_idx), train_valid_len)
    
    best_val = np.inf
    clip = 0
    loss = []
    for epoch in range(epoch_n):
        try:
            torch.cuda.empty_cache()
        except:
            pass
            
        optimizer.zero_grad()
        output1= model(feature.float(), embed_feature.float(), adj)
        '''
        pcc_loss_train = torch.mean(pcc_torch(output1[list(np.array(train_idx)+test_len)], label[list(np.array(train_idx)+test_len)].float(), batch_first=False))
        pcc_loss_val = torch.mean(pcc_torch(output1[list(np.array(valid_idx)+test_len)], label[list(np.array(valid_idx)+test_len)].float(), batch_first=False))
        '''
        loss_train1 = loss_fn(output1[list(np.array(train_idx)+test_len)], label[list(np.array(train_idx)+test_len)].float())
        loss_val1 = loss_fn(output1[list(np.array(valid_idx)+test_len)], label[list(np.array(valid_idx)+test_len)].float())
        if load_test_groundtruth == True: 
            #pcc_loss_test = torch.mean(pcc_torch(output1[:test_len], label[:test_len].float(), batch_first=False))
            loss_test1 = loss_fn(output1[:test_len], label[:test_len].float())
            loss.append([loss_train1.item(), loss_val1.item(), loss_test1.item()])
        else:
            loss.append([loss_train1.item(), loss_val1.item(), None])
        
        if epoch % print_epoch_step == 0:
            print("******************************************")
            print("Epoch {}/{}".format(epoch+1, epoch_n),
                  'loss_train: {:.4f}'.format(loss_train1.item()),
                  'loss_val: {:.4f}'.format(loss_val1.item()),
                  end = '\t'
                 )
            if load_test_groundtruth == True:
                print("Test loss= {:.4f}".format(loss_test1.item()), end = '\t')
            print('time: {:.4f}s'.format(time.time() - time_open))
        
        if early_stopping_patience > 0:
            if torch.round(loss_val1*10000)/10000 < best_val:
                best_val = torch.round(loss_val1*10000)/10000
                best_loss = loss.copy()
                clip = 1
            else:
                clip += 1
                if clip == early_stopping_patience:
                    break
        else:
            best_loss = loss.copy()
        
        loss_train1.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
        optimizer.step()
        if scheduler != None:
            try:
                scheduler.step()
            except:
                scheduler.step(metrics = loss_val1)
                
    print("***********************Final Loss***********************")
    print("Epoch {}/{}".format(epoch+1, epoch_n),
          'loss_train: {:.4f}'.format(loss_train1.item()),
          'loss_val: {:.4f}'.format(loss_val1.item()),
          end = '\t'
         )
    if load_test_groundtruth == True:
        print("Test loss= {:.4f}".format(loss_test1.item()), end = '\t')
    print('time: {:.4f}s'.format(time.time() - time_open))
    
    torch.cuda.empty_cache()
        
    return output1.cpu(), loss, model.cpu()
class conGCN(nn.Module):
    def __init__(self, nfeat, 
                 nembed,
                 nhid, 
                 common_hid_layers_num, 
                 fcnn_hid_layers_num, 
                 dropout, 
                 nout1, 
                 coord,
                 use_embedfeature #此处有修改代码，是否使用embedding
                ):
        super(conGCN, self).__init__()

        self.nfeat = nfeat
        #加入embed
        self.nembed = nembed
        self.nhid = nhid
        self.common_hid_layers_num = common_hid_layers_num
        self.fcnn_hid_layers_num = fcnn_hid_layers_num
        self.nout1 = nout1
        self.dropout = dropout
        self.training = True
        self.coord = coord
        self.use_embedfeature = use_embedfeature
        ## The beginning layer
        self.gc_in_exp = conGraphConvolutionlayer(nfeat, nhid)
        self.bn_node_in_exp = nn.BatchNorm1d(nhid)
        self.gc_in_sp = conGraphConvolutionlayer(nfeat, nhid)
        self.bn_node_in_sp = nn.BatchNorm1d(nhid)
        if self.use_embedfeature == True:
            self.gc_in_embed = conGraphConvolutionlayer(nembed, nhid)
            self.bn_node_in_embed = nn.BatchNorm1d(nhid)
        
        ## common_hid_layers
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec('self.cgc{}_exp = conGraphConvolutionlayer(nhid, nhid)'.format(i+1))
                exec('self.bn_node_chid{}_exp = nn.BatchNorm1d(nhid)'.format(i+1))
                exec('self.cgc{}_sp = conGraphConvolutionlayer(nhid, nhid)'.format(i+1))
                exec('self.bn_node_chid{}_sp = nn.BatchNorm1d(nhid)'.format(i+1))
                # 加入embedding
                if self.use_embedfeature == True:
                    exec('self.cgc{}_embed = conGraphConvolutionlayer(nhid, nhid)'.format(i+1))
                    exec('self.bn_node_chid{}_embed = nn.BatchNorm1d(nhid)'.format(i+1))
                
        ## FCNN layers
        #使用coord信息，有改动
        # 是否使用embedding信息
        if self.coord == True:
            if self.use_embedfeature == True:
                self.gc_out11 = nn.Linear(3*nhid, nhid, bias=True)
            else:
                self.gc_out11 = nn.Linear(2*nhid, nhid, bias=True)
        else:
            if self.use_embedfeature == True:
                self.gc_out11 = nn.Linear(2*nhid, nhid, bias=True)
            else:
                self.gc_out11 = nn.Linear(nhid, nhid, bias=True)
            
        self.bn_out1 = nn.BatchNorm1d(nhid)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec('self.gc_out11{} = nn.Linear(nhid, nhid, bias=True)'.format(i+1))
                exec('self.bn_out11{} = nn.BatchNorm1d(nhid)'.format(i+1))
        self.gc_out12 = nn.Linear(nhid, nout1, bias=True)
        

    def forward(self, x, embed, adjs):    
        
        self.x = x
        self.embed = embed
        
        ## input layer
        self.x_exp = self.gc_in_exp(self.x, adjs[0])
        self.x_exp = self.bn_node_in_exp(self.x_exp)
        self.x_exp = F.elu(self.x_exp)
        self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
        self.x_sp = self.gc_in_sp(self.x, adjs[1])
        self.x_sp = self.bn_node_in_sp(self.x_sp)
        self.x_sp = F.elu(self.x_sp)
        self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        # embedding
        if self.use_embedfeature == True:
            self.x_embed = self.gc_in_embed(self.embed, adjs[2])
            self.x_embed = self.bn_node_in_embed(self.x_embed)
            self.x_embed = F.elu(self.x_embed)
            self.x_embed = F.dropout(self.x_embed, self.dropout, training=self.training)

        #修改到这里
        ## common layers
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("self.x_exp = self.cgc{}_exp(self.x_exp, adjs[0])".format(i+1))
                exec("self.x_exp = self.bn_node_chid{}_exp(self.x_exp)".format(i+1))
                self.x_exp = F.elu(self.x_exp)
                self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
                exec("self.x_sp = self.cgc{}_sp(self.x_sp, adjs[1])".format(i+1))
                exec("self.x_sp = self.bn_node_chid{}_sp(self.x_sp)".format(i+1))
                self.x_sp = F.elu(self.x_sp)
                self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)
                # embedding
                if self.use_embedfeature == True:
                    exec("self.x_embed = self.cgc{}_embed(self.x_embed, adjs[2])".format(i+1))
                    exec("self.x_embed = self.bn_node_chid{}_embed(self.x_embed)".format(i+1))
                    self.x_embed = F.elu(self.x_embed)
                    self.x_embed = F.dropout(self.x_embed, self.dropout, training=self.training)

        ## FCNN layers
        if self.coord == True:
            if self.use_embedfeature == True:
                self.x1 = torch.cat([self.x_exp, self.x_sp, self.x_embed], dim=1)
            else:
                self.x1 = torch.cat([self.x_exp, self.x_sp], dim=1)
        else:
            if self.use_embedfeature == True:
                self.x1 = torch.cat([self.x_exp, self.x_embed], dim=1)
            else:
                self.x1 = self.x_exp
                
        self.x1 = self.gc_out11(self.x1)
        self.x1 = self.bn_out1(self.x1)
        self.x1 = F.elu(self.x1)
        self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec("self.x1 = self.gc_out11{}(self.x1)".format(i+1))
                exec("self.x1 = self.bn_out11{}(self.x1)".format(i+1))
                self.x1 = F.elu(self.x1)
                self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        self.x1 = self.gc_out12(self.x1)

        gc_list = {}
        gc_list['gc_in_exp'] = self.gc_in_exp
        gc_list['gc_in_sp'] = self.gc_in_sp
        if self.use_embedfeature == True:
            gc_list['gc_in_embed'] = self.gc_in_embed
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("gc_list['cgc{}_exp'] = self.cgc{}_exp".format(i+1, i+1))
                exec("gc_list['cgc{}_sp'] = self.cgc{}_sp".format(i+1, i+1))
                if self.use_embedfeature == True:
                    exec("gc_list['cgc{}_embed'] = self.cgc{}_embed".format(i+1, i+1))
        gc_list['gc_out11'] = self.gc_out11
        if self.fcnn_hid_layers_num > 0:
            exec("gc_list['gc_out11{}'] =  self.gc_out11{}".format(i+1, i+1))
        gc_list['gc_out12'] = self.gc_out12
        
        return F.log_softmax(self.x1, dim=1), gc_list



def conGCN_train(model, 
                 train_valid_len,
                 test_len, 
                 feature, 
                 embed_feature, #加入embed
                 adjs, 
                 label, 
                 epoch_n, 
                 loss_fn, 
                 optimizer, 
                 train_valid_ratio = 0.9,
                 scheduler = None,
                 early_stopping_patience = 5,
                 clip_grad_max_norm = 1,
                 load_test_groundtruth = False,
                 print_epoch_step = 1,
                 cpu_num = -1,
                 GCN_device = 'CPU',
                ):
    
    if GCN_device == 'CPU':
        device = torch.device("cpu")
        print('Use CPU as device.')
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('Use GPU as device.')
        else:
            device = torch.device("cpu")
            print('Use CPU as device.')
    
    if cpu_num == -1:
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
    else:
        torch.set_num_threads(cpu_num)
    
    model = model.to(device)
    adjs = [adj.to(device) for adj in adjs]
    feature = feature.to(device)
    # embed 加入
    embed_feature = embed_feature.to(device)
    label = label.to(device)
    
    time_open = time.time()

    train_idx = range(int(train_valid_len*train_valid_ratio))
    valid_idx = range(len(train_idx), train_valid_len)
    
    best_val = np.inf
    clip = 0
    loss = []
    para_list = []
    for epoch in range(epoch_n):
        try:
            torch.cuda.empty_cache()
        except:
            pass
            
        optimizer.zero_grad()
        output1, paras = model(feature.float(), embed_feature.float(), adjs)
        '''
        pcc_loss_train = torch.mean(pcc_torch(output1[list(np.array(train_idx)+test_len)], label[list(np.array(train_idx)+test_len)].float(), batch_first=False))
        pcc_loss_val = torch.mean(pcc_torch(output1[list(np.array(valid_idx)+test_len)], label[list(np.array(valid_idx)+test_len)].float(), batch_first=False))
        '''
        loss_train1 = loss_fn(output1[list(np.array(train_idx)+test_len)], label[list(np.array(train_idx)+test_len)].float())
        loss_val1 = loss_fn(output1[list(np.array(valid_idx)+test_len)], label[list(np.array(valid_idx)+test_len)].float())
        if load_test_groundtruth == True: 
            #pcc_loss_test = torch.mean(pcc_torch(output1[:test_len], label[:test_len].float(), batch_first=False))
            loss_test1 = loss_fn(output1[:test_len], label[:test_len].float())
            loss.append([loss_train1.item(), loss_val1.item(), loss_test1.item()])
        else:
            loss.append([loss_train1.item(), loss_val1.item(), None])
        
        if epoch % print_epoch_step == 0:
            print("******************************************")
            print("Epoch {}/{}".format(epoch+1, epoch_n),
                  'loss_train: {:.4f}'.format(loss_train1.item()),
                  'loss_val: {:.4f}'.format(loss_val1.item()),
                  end = '\t'
                 )
            if load_test_groundtruth == True:
                print("Test loss= {:.4f}".format(loss_test1.item()), end = '\t')
            print('time: {:.4f}s'.format(time.time() - time_open))
        para_list.append(paras.copy())
        for i in paras.keys():
            para_list[-1][i] = copy.deepcopy(para_list[-1][i])
        
        if early_stopping_patience > 0:
            if torch.round(loss_val1, decimals=4) < best_val:
                best_val = torch.round(loss_val1, decimals=4)
                best_paras = paras.copy()
                best_loss = loss.copy()
                clip = 1
                for i in paras.keys():
                    best_paras[i] = copy.deepcopy(best_paras[i])
            else:
                clip += 1
                if clip == early_stopping_patience:
                    break
        else:
            best_loss = loss.copy()
            best_paras = None
        
        loss_train1.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
        optimizer.step()
        if scheduler != None:
            try:
                scheduler.step()
            except:
                scheduler.step(metrics = loss_val1)
                
    print("***********************Final Loss***********************")
    print("Epoch {}/{}".format(epoch+1, epoch_n),
          'loss_train: {:.4f}'.format(loss_train1.item()),
          'loss_val: {:.4f}'.format(loss_val1.item()),
          end = '\t'
         )
    if load_test_groundtruth == True:
        print("Test loss= {:.4f}".format(loss_test1.item()), end = '\t')
    print('time: {:.4f}s'.format(time.time() - time_open))
    
    torch.cuda.empty_cache()
        
    return output1.cpu(), loss, model.cpu()
