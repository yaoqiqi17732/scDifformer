import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from collections import Counter
import os
import sys
import torch
import matplotlib.pyplot as plt
import scanpy as sc
import pickle
import warnings
import torch
from STdGCN.__init__ import *
from sklearn.preprocessing import StandardScaler
from STdGCN.benchmark import compare_results
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--simulation_number', type=int, default=27, help='the number of dataset')
parser.add_argument('--use_marker_genes', type=bool, default=True, help='whether use marker genes')
parser.add_argument('--generate_new_pseudo_spots', type=bool, default=False, help='whether generate new pseudo spots')
parser.add_argument('--external_genes', type=bool, default=True, help='whether exists external genes')
parser.add_argument('--use_embedfeature', type=bool, default=False, help='whether use embedfeature')
parser.add_argument('--calculate_adj', type=bool, default=False, help='whether calculate adjacent matrix')
parser.add_argument('--front_embed', type=bool, default=False, help='whether to use front embed')
parser.add_argument('--final_output_path', type=str, default='/min4max16_results/50validation_results/', help='the final output path')
parser.add_argument('--nhid', type=int, default=256, help='the dimension of the hidden layer')
args = parser.parse_args()

#total_simulate_list = [11, 12, 13, 19, 20, 21, 22]
#for simulation_number in [21]:

simulation_number = args.simulation_number
front_embed = args.front_embed
final_output_path = args.final_output_path
nhid = args.nhid

data_path = '/mnt/nfs/SimualtedSpatalData/'
scRNA = sc.read_h5ad(data_path + 'dataset' + str(simulation_number) + '/scRNA.h5ad')
Spatial = sc.read_h5ad(data_path + 'dataset' + str(simulation_number) + '/Spatial.h5ad')
total_genes = list(set(scRNA.var_names).intersection(set(Spatial.var_names)))
scRNA = scRNA[:, total_genes]
Spatial = Spatial[:, total_genes]

print('The shape of scRNA:', scRNA.X.shape)
print('The shape of Spatial:', Spatial.X.shape)

n_jobs = -1
load_test_groundtruth = True
GCN_device = 'GPU'
fraction_pie_plot = False
cell_type_distribution_plot = False
use_marker_genes = args.use_marker_genes
generate_new_pseudo_spots = args.generate_new_pseudo_spots
external_genes = args.external_genes
output_path = '/mnt/nfs/wbzhang/stdgcn/data/simulate/dataset' + str(simulation_number)
use_embedfeature = args.use_embedfeature
calculate_adj = args.calculate_adj

# sc_data
sc_adata = scRNA
sc_adata.obs.rename(columns={'celltype_final': 'cell_type'}, inplace=True)
cell_type_num = len(sc_adata.obs['cell_type'].unique())
print('cell_type_num:', cell_type_num)
cell_types = sc_adata.obs['cell_type'].unique()

word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}
idx_to_word_celltype = {i: word for i, word in enumerate(cell_types)}

celltype_idx = [word_to_idx_celltype[w] for w in sc_adata.obs['cell_type']]
sc_adata.obs['cell_type_idx'] = celltype_idx
sc_adata.obs['cell_type'].value_counts()

find_marker_genes_paras = {
    'preprocess': True,
    'normalize': True,
    'log': True,
    'highly_variable_genes': False,
    'highly_variable_gene_num': None,
    'regress_out': False,
    'PCA_components': 30, 
    'marker_gene_method': 'logreg',
    'top_gene_per_type': 100,
    'filter_wilcoxon_marker_genes': True,
    'pvals_adj_threshold': 0.10,
    'log_fold_change_threshold': 1,
    'min_within_group_fraction_threshold': None,
    'max_between_group_fraction_threshold': None,
}

if use_marker_genes == True:
    if external_genes == True:
        with open(output_path+"/marker_genes.tsv", 'r') as f:
            selected_genes = [line.rstrip('\n') for line in f]
    else:
        selected_genes, cell_type_marker_genes = find_marker_genes(sc_adata,
                                                                  preprocess = find_marker_genes_paras['preprocess'],
                                                                  highly_variable_genes = find_marker_genes_paras['highly_variable_genes'],
                                                                  PCA_components = find_marker_genes_paras['PCA_components'], 
                                                                  filter_wilcoxon_marker_genes = find_marker_genes_paras['filter_wilcoxon_marker_genes'], 
                                                                  marker_gene_method = find_marker_genes_paras['marker_gene_method'],
                                                                  pvals_adj_threshold = find_marker_genes_paras['pvals_adj_threshold'],
                                                                  log_fold_change_threshold = find_marker_genes_paras['log_fold_change_threshold'],
                                                                  min_within_group_fraction_threshold = find_marker_genes_paras['min_within_group_fraction_threshold'],
                                                                  max_between_group_fraction_threshold = find_marker_genes_paras['max_between_group_fraction_threshold'],
                                                                  top_gene_per_type = find_marker_genes_paras['top_gene_per_type'])
        with open(output_path+"/marker_genes.tsv", 'w') as f:
            for gene in selected_genes:
                f.write(str(gene) + '\n')
        
print("{} genes have been selected as marker genes.".format(len(selected_genes)))

'''
downsample: 是否使用航姐downsample的方法
'''

pseudo_spot_simulation_paras = {
    'spot_num': 30000,
    'min_cell_num_in_spot': 4,
    'max_cell_num_in_spot': 16,
    'generation_method': 'celltype',
    'max_cell_types_in_spot': 4,
    'downsample': False
}
if generate_new_pseudo_spots == True:
    pseudo_adata = pseudo_spot_generation(sc_adata,
                                          idx_to_word_celltype,
                                          spot_num = pseudo_spot_simulation_paras['spot_num'],
                                          min_cell_number_in_spot = pseudo_spot_simulation_paras['min_cell_num_in_spot'],
                                          max_cell_number_in_spot = pseudo_spot_simulation_paras['max_cell_num_in_spot'],
                                          max_cell_types_in_spot = pseudo_spot_simulation_paras['max_cell_types_in_spot'],
                                          generation_method = pseudo_spot_simulation_paras['generation_method'],
                                          n_jobs = n_jobs,
                                          downsample = pseudo_spot_simulation_paras['downsample']
                                          )
    data_file = open(output_path+'/min4max16_pseudo_ST.pkl','wb')
    pickle.dump(pseudo_adata, data_file)
    data_file.close()
else:
    data_file = open(output_path+'/min4max16_pseudo_ST.pkl','rb')
    pseudo_adata = pickle.load(data_file)
    data_file.close()

data_normalization_paras = {
    'normalize': True, 
    'log': True, 
    'scale': False,
}
dfsum = Spatial.uns['density'].sum(axis=1)
ST_adata = Spatial
ST_groundtruth = Spatial.uns['density'].div(dfsum, axis='rows')
if load_test_groundtruth == True:
    for i in cell_types:
        ST_adata.obs[i] = np.array(ST_groundtruth[i])
        
ST_adata_filter = ST_adata
pseudo_adata_filter = pseudo_adata
ST_adata_filter_norm = ST_preprocess(ST_adata_filter,
                                     normalize = data_normalization_paras['normalize'], 
                                     log = data_normalization_paras['log'], 
                                     scale = data_normalization_paras['scale'],
                                    )[:, list(selected_genes)]

pseudo_adata_norm = ST_preprocess(pseudo_adata_filter, 
                                  normalize = data_normalization_paras['normalize'], 
                                  log = data_normalization_paras['log'], 
                                  scale = data_normalization_paras['scale'],
                                 )[:, list(selected_genes)]
try:
    try:
        ST_adata_filter_norm.obs.insert(0, 'cell_num', ST_adata_filter.obs['cell_num'])
    except:
        ST_adata_filter_norm.obs['cell_num'] = ST_adata_filter.obs['cell_num']
except:
    ST_adata_filter_norm.obs.insert(0, 'cell_num', [0]*ST_adata_filter_norm.obs.shape[0])
for i in cell_types:
    try:
        ST_adata_filter_norm.obs[i] = ST_adata_filter.obs[i]
    except:
        ST_adata_filter_norm.obs[i] = [0]*ST_adata_filter_norm.obs.shape[0]
try:
    ST_adata_filter_norm.obs['cell_type_num'] = (ST_adata_filter_norm.obs[cell_types]>0).sum(axis=1)
except:
    ST_adata_filter_norm.obs['cell_type_num'] = [0]*ST_adata_filter_norm.obs.shape[0]
    
pseudo_adata_norm.obs['cell_type_num'] = (pseudo_adata_norm.obs[cell_types]>0).sum(axis=1)


if calculate_adj == True:
    ST_Adj = interADJ(ST_adata_filter_norm, k=10, distanceType='euclidean', pruneTag='NA')
    print('ST_Adj:', ST_Adj.shape)
    pseudo_Adj = interADJ(pseudo_adata_norm, k=20, distanceType='euclidean', pruneTag='NA')
    print('pseudo_Adj:', pseudo_Adj.shape)
    ST_pseudo_Adj = exterADJ(ST_adata_filter_norm, pseudo_adata_norm, k=20, distanceType='euclidean', pruneTag='NA')
    print('ST_pseudo_Adj:', ST_pseudo_Adj.shape)
    
    os.makedirs(output_path + '/min4max16_ADJ', exist_ok=True)
    torch.save(ST_Adj, output_path + '/min4max16_ADJ/ST_Adj.pth')
    torch.save(pseudo_Adj, output_path + '/min4max16_ADJ/pseudo_Adj.pth')
    torch.save(ST_pseudo_Adj, output_path + '/min4max16_ADJ/ST_pseudo_Adj.pth')
else:
    ST_Adj = torch.load(output_path + '/min4max16_ADJ/ST_Adj.pth')
    print('ST_Adj:', ST_Adj.shape)
    pseudo_Adj = torch.load(output_path + '/min4max16_ADJ/pseudo_Adj.pth')
    print('pseudo_Adj:', pseudo_Adj.shape)
    ST_pseudo_Adj = torch.load(output_path + '/min4max16_ADJ/ST_pseudo_Adj.pth')
    print('ST_pseudo_Adj:', ST_pseudo_Adj.shape)

real_num = ST_adata_filter_norm.shape[0]
pseudo_num = pseudo_adata.shape[0]

adj_real_pseudo = ADJ_transfer(ST_pseudo_Adj, 'real_pseudo', real_num, pseudo_num)
adj_real = ADJ_transfer(ST_Adj, 'real', real_num, pseudo_num)
adj_pseudo = ADJ_transfer(pseudo_Adj, 'pseudo', real_num, pseudo_num)

adj_alpha = 1
adj_beta = 1
diag_power = 20
adj_balance = (1+adj_alpha+adj_beta)*diag_power
adj_exp = torch.tensor(adj_real_pseudo+adj_alpha*adj_pseudo+adj_beta*adj_real)/adj_balance + torch.eye(adj_real_pseudo.shape[0])

norm = True
if norm == True:
    adj_exp = torch.tensor(adj_normalize(adj_exp, symmetry=True))

integration_for_feature_paras = {
    'batch_removal_method': None,
    'dimensionality_reduction_method': None,
    'dim': 80,
    'scale': True,
}

ST_integration_batch_removed = data_integration(ST_adata_filter_norm, 
                                                pseudo_adata_norm, 
                                                batch_removal_method=integration_for_feature_paras['batch_removal_method'], 
                                                dim=min(int(ST_adata_filter_norm.shape[1]*1/2), integration_for_feature_paras['dim']), 
                                                dimensionality_reduction_method=integration_for_feature_paras['dimensionality_reduction_method'], 
                                                scale=integration_for_feature_paras['scale'],
                                                cpu_num=n_jobs,
                                                AE_device=GCN_device
                                               )
feature = torch.tensor(ST_integration_batch_removed.iloc[:, 3:].values)
print('The expression feature:', feature.shape)

# 加入embedding
if use_embedfeature == True:
    data_file = open(output_path+'/embedding_diffusion/sc/24L_pseudo_ST.pkl','rb')
    embed_pseudo_adata = pickle.load(data_file)
    data_file.close()
    #pseudo_adata_embed = torch.load('/mnt/nfs/wbzhang/stdgcn/data/simulate/dataset' + str(simulation_number) + '/embedding_diffusion/pseudo/max4_pca512_embed.pth').numpy()
    pseudo_adata_embed = embed_pseudo_adata.X
    ST_adata_embed = torch.load('/mnt/nfs/wbzhang/stdgcn/data/simulate/dataset'+str(simulation_number)+'/embedding_diffusion/ST/24L_scST.pth')
    embed_feature = torch.tensor(np.concatenate((ST_adata_embed, pseudo_adata_embed),axis=0))
    print('The embedding feature:', embed_feature.shape)
else:
    embed_feature = torch.Tensor(1)

if front_embed == True:
    #将embedding数据放在前面 一起PCA的数据
    data_file = open(output_path+'/embedding_diffusion/sc/12L_pseudo_ST.pkl','rb')
    embed_pseudo_adata = pickle.load(data_file)
    data_file.close()
    #pseudo_adata_embed = torch.load('/mnt/nfs/wbzhang/stdgcn/data/simulate/dataset' + str(simulation_number) + '/embedding_diffusion/pseudo/max4_pca512_embed.pth').numpy()
    pseudo_adata_embed = embed_pseudo_adata.X
    ST_adata_embed = torch.load('/mnt/nfs/wbzhang/stdgcn/data/simulate/dataset'+str(simulation_number)+'/embedding_diffusion/ST/12L_scST.pth')
    embed_feature = torch.tensor(np.concatenate((ST_adata_embed, pseudo_adata_embed),axis=0))
    
    #是否要用scale 将范围控制到0以上
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 3))
    scaled = scaler.fit_transform(embed_feature.numpy())
    embed_feature = torch.tensor(scaled)
    
    print('The embedding feature:', embed_feature.shape)
    feature = torch.concatenate((feature, embed_feature), axis=1)
    print('Total shape:', feature.shape)
else:
    pass

for i in range(50):
    GCN_paras = {
        'epoch_n': 3000,
        'dim': 80,
        'gcn_hid_layers_num': 2,
        'fcnn_hid_layers_num': 1,
        'dropout': 0,
        'learning_rate_SGD': 2e-1,
        'weight_decay_SGD': 3e-4,
        'momentum': 0.9,
        'dampening': 0,
        'nesterov': True,
        'early_stopping_patience': 20,
        'clip_grad_max_norm': 1,
        #'LambdaLR_scheduler_coefficient': 0.997,
        'print_loss_epoch_step': 20,
    }
    input_layer = feature.shape[1]
    # 加入embedding
    if use_embedfeature == True:
        embed_input_layer = embed_feature.shape[1]
    else:
        embed_input_layer = None
    hidden_layer = min(int(ST_adata_filter_norm.shape[1]*1/2), GCN_paras['dim'])
    output_layer1 = len(word_to_idx_celltype)
    epoch_n = GCN_paras['epoch_n']
    gcn_hid_layers_num = GCN_paras['gcn_hid_layers_num']
    fcnn_hid_layers_num = GCN_paras['fcnn_hid_layers_num']
    dropout = GCN_paras['dropout']
    learning_rate_SGD = GCN_paras['learning_rate_SGD']
    weight_decay_SGD = GCN_paras['weight_decay_SGD']
    momentum = GCN_paras['momentum']
    dampening = GCN_paras['dampening']
    nesterov = GCN_paras['nesterov']
    early_stopping_patience = GCN_paras['early_stopping_patience']
    clip_grad_max_norm = GCN_paras['clip_grad_max_norm']
    LambdaLR_scheduler_coefficient = 0.997
    ReduceLROnPlateau_factor = 0.1
    ReduceLROnPlateau_patience = 5
    scheduler = 'scheduler_ReduceLROnPlateau'
    print_epoch_step = GCN_paras['print_loss_epoch_step']
    cpu_num = n_jobs
    
    model = GraphSAGE(nfeat = input_layer, 
                      nfeat_embed = embed_input_layer,
                      nhid = nhid,
                      nhid_embed = 32,
                      gcn_hid_layers_num = gcn_hid_layers_num,
                      fcnn_hid_layers_num = fcnn_hid_layers_num,
                      dropout = dropout, 
                      nout1 = output_layer1,
                      training = True,
                      policy = 'mean',
                      gcn = True,
                      use_embedfeature = use_embedfeature
                     )
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr = learning_rate_SGD, 
                                momentum = momentum, 
                                weight_decay = weight_decay_SGD, 
                                dampening = dampening, 
                                nesterov = nesterov)
    
    scheduler_LambdaLR = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                           lr_lambda = lambda epoch: LambdaLR_scheduler_coefficient ** epoch)
    scheduler_ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                             mode='min', 
                                                                             factor=ReduceLROnPlateau_factor, 
                                                                             patience=ReduceLROnPlateau_patience, 
                                                                             threshold=0.0001, 
                                                                             threshold_mode='rel', 
                                                                             cooldown=0, 
                                                                             min_lr=0)
    if scheduler == 'scheduler_LambdaLR':
        scheduler = scheduler_LambdaLR
    elif scheduler == 'scheduler_ReduceLROnPlateau':
        scheduler = scheduler_ReduceLROnPlateau
    else:
        scheduler = None
    
    loss_fn1 = nn.KLDivLoss(reduction = 'mean')
    
    train_valid_len = pseudo_adata.shape[0]
    test_len = ST_adata_filter.shape[0]
    
    table1 = ST_adata_filter_norm.obs.copy()
    label1 = pd.concat([table1[pseudo_adata.obs.iloc[:,:-1].columns], pseudo_adata.obs.iloc[:,:-1]])
    label1 = torch.tensor(label1.values)
    adj = adj_exp.float()
    
    output1, loss, trained_model = GraphSAGE_train(model = model, 
                                                   train_valid_len = train_valid_len,
                                                   train_valid_ratio = 0.9,
                                                   test_len = test_len, 
                                                   feature = feature, 
                                                   embed_feature = embed_feature,
                                                   adj = adj, 
                                                   label = label1, 
                                                   epoch_n = epoch_n, 
                                                   loss_fn = loss_fn1, 
                                                   optimizer = optimizer, 
                                                   scheduler = scheduler, 
                                                   early_stopping_patience = early_stopping_patience,
                                                   clip_grad_max_norm = clip_grad_max_norm,
                                                   load_test_groundtruth = load_test_groundtruth,
                                                   print_epoch_step = print_epoch_step,
                                                   cpu_num = cpu_num,
                                                   GCN_device = GCN_device
                                                  )
    
    #final_path = output_path + '/embedding_diffusion_results/min4max16_results/more_tests'
    #final_path = output_path + '/min4max16_results/30validation_results/' + str(i)
    #final_path = output_path + '/embedding_diffusion_results/front_min4max16_results/min4max16_results/30validation_results/' + str(i)
    #final_path = output_path + '/embedding_diffusion_results/min4max16_results/30validation_results/' + str(i)
    #final_path = output_path + '/embedding_diffusion_results/front_min4max16_results/50validation_results/'
    final_path = output_path + final_output_path + str(i)
    os.makedirs(final_path, exist_ok=True)
    
    loss_table = pd.DataFrame(loss, columns=['train', 'valid', 'test'])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(loss_table.index, loss_table['train'], label='train')
    ax.plot(loss_table.index, loss_table['valid'], label='valid')
    if load_test_groundtruth == True:
        ax.plot(loss_table.index, loss_table['test'], label='test')
    ax.set_xlabel('Epoch', fontsize = 20)
    ax.set_ylabel('Loss', fontsize = 20)
    ax.set_title('Loss function curve', fontsize = 20)
    ax.legend(fontsize = 15)
    plt.tight_layout()
    plt.savefig(final_path+'/Loss_function.jpg', dpi=300)
    plt.close('all')
    
    predict_table = pd.DataFrame(np.exp(output1[:test_len].detach().numpy()).tolist(), index=ST_adata_filter_norm.obs.index, columns=pseudo_adata_norm.obs.columns[:-2])
    predict_table.to_csv(final_path+'/predict_result.csv', index=True, header=True)
    
    torch.save(trained_model, final_path+'/model_parameters')
    
    pred_use = np.round_(output1.exp().detach()[:test_len], decimals=4)
    cell_type_list = cell_types
    #coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]
    
    if fraction_pie_plot == True:
        plot_frac_results(pred_use, cell_type_list, coordinates, point_size=300, size_coefficient=0.0009, file_name=output_path+'/predict_results_pie_plot.jpg', if_show=False)
        
    if cell_type_distribution_plot == True:
        plot_scatter_by_type(pred_use, cell_type_list, coordinates, point_size=300, file_path=output_path, if_show=False)
    
    ST_adata_filter_norm.obsm['predict_result'] = np.exp(output1[:test_len].detach().numpy())
    
    torch.cuda.empty_cache()
    
    prediction = pd.read_csv(final_path + '/predict_result.csv', index_col=0)
    methods = ['pcc', 'mae', 'jsd', 'rmse', 'ssim']
    compare_result = pd.DataFrame(None)
    for method in methods:
        a = compare_results(ST_adata.obs[cell_types], prediction, metric=method, columns=[method], axis=1)
        compare_result = pd.concat([compare_result, a], axis=1)
    col_mean = compare_result.mean(axis=0)
    col_mean.name = 'mean'
    df = compare_result.append(col_mean)
    df.to_csv(final_path+'/min4max16_benchmark.csv', index=True, header=True)