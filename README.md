# RumorGuard

**NOTE:** For Supplementary material, refer to `supplementary_material.pdf`.


## Requirements

Install requirements by:

```
pip install -r requirements.txt
```

## resources

We provide the datasets and GNN models used in the experiment. For convenience, you can refer to these resources in the code through the following variables.

- train graphs : graph_WC_train, graph_CL_train, graph_ET_train
- test graphs : graph_WC_test, graph_CL_test, graph_ET_test
- train datasets : WC1000, CL1000, ET1000
- test datasets : WC50, CL50, ET50
- trained 2-layer GCN models : model_WC_2GCN, model_CL_2GCN, model_ET_2GCN
- trained 6-layer GCN models : model_WC_6GCN, model_CL_6GCN, model_ET_6GCN

## Experiments

You can conduct the following experiments in a single file experiments.ipynb

### 1. Generate dataset

Generate a taining dataset of size data_num by generate_dataset function in dataset.ipynb

```
generate_dataset(graph_name, data_num, saving_tag)
```

By the same way, generate a test dataset.

for example,

```
generate_dataset(graph_ET_train, 1000, saving_tag='-1000')  # train dataset
generate_dataset(graph_ET_test, 50, saving_tag='-50')       # test dataset
```

### 2. Training

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
train(ET1000, ET50, saving_name='ET_2GCN.pt', hyper_params={'lr':0.05, 'lr_gamma':0.995, 'max_epoch':500, 'gnn_latent_dim':[128,128]}, gpu_num=0)
```

or you can do the hyperparameter tuning by Optuna:

```
hparam_tuning(train_dataset_name, test_dataset_name, saving_name, gpu_num)
```

you can evaluate influence estimation quality of GNN models by

```
evaluate_quality(model_name, gnn_latent_dim, train_dataset_name, gpu_num)
```

### 3. Evaluation

run rumor blocking (or baseline) algorithms by pipeline function in algorithm_pipeline.ipynb

```
pipeline(alg_name, dataset_name, del_edge_num, **alg_kwargs):
```

`alg_name` and corresponding `**alg_kwargs` can be

- ‘random’, ‘outdegree’, ‘betweenness’, ‘pagerank’, ‘KED’, ‘MDS’
    - no extra parameter
- ‘BPM’, ‘MBPM’
    - sampling_num = sampling number of bond percolation method (default=10000)
- ‘RIS’
    - eps = error range (default=0.2)
- ‘greedy’
    - simul_num = number of MC simulation for each step.
- ‘RumorGuard_G’, ‘RumorGuard_I’
    - model_name
    - gnn_latent_dim
    - gpu_num
- ‘RumorGuard_O’
    - model_name
    - gnn_latent_dim
    - gpu_num
    - epochs, lr, lr_gamma, size_coeff, ent_coeff = GNN optimizing parameters
        - for convenience, you can use **default_hyper_params
