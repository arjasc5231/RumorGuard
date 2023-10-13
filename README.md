# RumorGuard



## Requirements

Install requirements by:

```
pip install -r requirements.txt
```

## Generate dataset

Generate a taining dataset of size data_num by generate_dataset function in dataset.ipynb

```
generate_dataset(graph_name, data_num, saving_tag)
```

By the same way, generate a test dataset.

for example,

```
generate_dataset('Extended_train_LP', 1000, saving_tag='-1000')  # train dataset
generate_dataset('Extended_train_LP', 50, saving_tag='-50')      # test dataset
```

## Training

Train GCN model by train function in train.ipynb

```
train(train_dataset_name, test_dataset_name, hyper_params, saving_name, gpu_num):
```

‘hyper_params’ can be a dictionary of hyperparameters (or None)

- ‘max_epoch’
- ‘lr’
- ‘lr_gamma’
- ‘gnn_latent_dim’

for example,

```
train('Extended_train_LP-1000.pkl.gz', 'Extended_train_LP-50.pkl.gz', saving_name='E_2GCN.pt', hyper_params={'lr':0.001, 'lr_gamma':0.9995, 'max_epoch':2000, 'gnn_latent_dim':[128,128]}, gpu_num=0)
```

or you can do the hyperparameter tuning by Optuna:

```
hparam_tuning(train_dataset_name, test_dataset_name, saving_name, gpu_num)
```

you can evaluate appoximation quality by

```
approximation_quality(model_name, gnn_latent_dim, train_dataset_name, gpu_num)
```

## evaluation

run rumor blocking (or baseline) algorithms by pipeline function in algorithm_pipeline.ipynb

```
pipeline(alg_name, dataset_name, del_edge_num, **alg_kwargs):
```
alg_name and **alg_kwargs can be

- ‘random’, ‘outdegree’, ‘betweenness’, ‘pagerank’
    - no extra parameter
- ‘KED’, ‘MDS’
    - no extra parameter
- ‘greedy_BPM’
    - sampling_num = ampling number of bond percolation method (default=10000)
- ‘greedy’
    - simul_num = number of MC simulation for each step.
- ‘greedy_GNN’, ‘Saliency’
    - model_name
    - gnn_latent_dim
    - gpu_num
- ‘GNNExplainer_variant’
    - model_name
    - gnn_latent_dim
    - gpu_num
    - epochs
    - lr, lr_gamma, size_coeff, ent_coeff : GNN optimizing parameters
