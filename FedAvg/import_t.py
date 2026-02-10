import torch
import copy
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.Data_Prepper import Data_Prepper
from utils.helper_functions import (compute_grad_update, flatten, unflatten, 
                                   norm, add_gradient_updates, add_update_to_model,
                                   train_model, evaluate, mask_grad_update_by_order)
from os.path import join as oj

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment parameters
N = 10  # Honest participants
A = 11  # Adversaries performing label flipping
E = 5   # Local epochs
use_reputation = True
use_sparsify = False
threshold = 0.1  # Reputation threshold

# Initialize args
args = {
    'optimizer_fn': torch.optim.SGD,
    'loss_fn': torch.nn.CrossEntropyLoss(),
    'lr': 0.01,
    'lr_decay': 0.99,
    'rounds': 100,
    'Gamma': 1.0,
    'alpha': 0.9,
    'batch_size': 32,
    'train_val_split_ratio': 0.8,
    'dataset': 'cifar10',
    'model_fn': CIFAR10_CNN,
    'n_participants': N + A,
    'sample_size_cap': 600 * (N + A),
    'attack': 'lf',
    'n_adversaries': A,
    'split': 'uniform'
}

print("Initializing Federated Learning with:")
print(f"- {N} honest participants")
print(f"- {A} adversaries performing label flipping (1→7)")

# Initialize data
print("\nData Split information for honest participants:")
data_prepper = Data_Prepper(
    args['dataset'], 
    train_batch_size=args['batch_size'],
    n_participants=N,
    sample_size_cap=600 * N,
    train_val_split_ratio=args['train_val_split_ratio'],
    device=device,
    args_dict=args
)

valid_loader = data_prepper.get_valid_loader()
test_loader = data_prepper.get_test_loader()
train_loaders = data_prepper.get_train_loaders(N, args['split'])
shard_sizes = data_prepper.shard_sizes

# Initialize adversaries
print("\nData Split information for adversaries:")
adv_data_prepper = Data_Prepper(
    args['dataset'],
    train_batch_size=args['batch_size'],
    n_participants=A,
    sample_size_cap=600 * A,
    train_val_split_ratio=args['train_val_split_ratio'],
    device=device,
    args_dict=args
)
adv_loaders = adv_data_prepper.get_train_loaders(A, 'uniform')
shard_sizes += adv_data_prepper.shard_sizes

shard_sizes = torch.tensor(shard_sizes).float()
relative_shard_sizes = torch.div(shard_sizes, torch.sum(shard_sizes))
print(f"\nShard sizes: {shard_sizes.tolist()}")

# Initialize server model
server_model = args['model_fn']().to(device)
D = sum(p.numel() for p in server_model.parameters())

# Initialize participants
participant_models = [copy.deepcopy(server_model) for _ in range(N)]
participant_optimizers = [args['optimizer_fn'](model.parameters(), lr=args['lr']) for model in participant_models]
participant_schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, args['lr_decay']) for optimizer in participant_optimizers]

# Initialize adversaries
adv_models = [copy.deepcopy(server_model) for _ in range(A)]
adv_optimizers = [args['optimizer_fn'](model.parameters(), lr=args['lr']) for model in adv_models]
adv_schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, args['lr_decay']) for optimizer in adv_optimizers]

# Initialize reputation
R_set = set(range(N + A))
rs = torch.ones(N + A, device=device) / (N + A)

# Trackers
adv_lf_perfs = defaultdict(list)
valid_perfs = defaultdict(list)
local_perfs = defaultdict(list)
rs_dict = []
qs_dict = []
r_threshold = []

# FL Training
print("\nStarting Federated Learning...")
for _round in range(args['rounds']):
    gradients = []
    
    # Adversarial Updates
    for i in range(A):
        loader = adv_loaders[i]
        model = adv_models[i]
        optimizer = adv_optimizers[i]
        scheduler = adv_schedulers[i]
        
        # Label flipping attack (1→7)
        model_before = copy.deepcopy(model)
        for epoch in range(E):
            for batch in loader:
                data, target = batch[0].to(device), batch[1].to(device)
                target = torch.where(target == 1, torch.tensor(7).to(device), target)
                
                optimizer.zero_grad()
                loss = args['loss_fn'](model(data), target)
                loss.backward()
                optimizer.step()
                scheduler.step()
        
        adv_grad = compute_grad_update(model_before, model)
        flattened = flatten(adv_grad)
        norm_value = norm(flattened) + 1e-7
        if norm_value > args['Gamma']:
            adv_grad = unflatten(torch.multiply(torch.tensor(args['Gamma']), torch.div(flattened, norm_value)), adv_grad)
        gradients.append(adv_grad)

    # Honest Updates
    for i in range(N):
        loader = train_loaders[i]
        model = participant_models[i]
        optimizer = participant_optimizers[i]
        scheduler = participant_schedulers[i]
        
        model_before = copy.deepcopy(model)
        model = train_model(model, loader, args['loss_fn'], optimizer, device, E, scheduler)
        gradient = compute_grad_update(model_before, model)
        flattened = flatten(gradient)
        norm_value = norm(flattened) + 1e-7
        if norm_value > args['Gamma']:
            gradient = unflatten(torch.multiply(torch.tensor(args['Gamma']), torch.div(flattened, norm_value)), gradient)
        gradients.append(gradient)

    # Server Aggregation
    aggregated_gradient = [torch.zeros_like(param) for param in server_model.parameters()]
    
    if use_reputation:
        if _round == 0:
            weights = relative_shard_sizes
        else:
            weights = rs
        
        flat_aggre_grad = flatten(aggregated_gradient)
        phis = torch.zeros(N + A, device=device)
        for i in range(N + A):
            if i in R_set:
                phis[i] = torch.nn.functional.cosine_similarity(
                    flatten(gradients[i]).unsqueeze(0),
                    flat_aggre_grad.unsqueeze(0)
                )
        
        rs = args['alpha'] * rs + (1 - args['alpha']) * phis
        mask = torch.zeros(N + A, device=device)
        for i in R_set:
            mask[i] = 1
        rs = rs * mask
        rs = rs / (rs.sum() + 1e-10)
        
        if _round >= 10 and len(R_set) > 1:
            curr_threshold = threshold * (1.0 / len(R_set))
            to_remove = [i for i in R_set if rs[i] < curr_threshold]
            
            for i in to_remove:
                R_set.remove(i)
                print(f"Round {_round}: Removed participant {i}")
            
            if not R_set:
                print("Stopping training - no participants remain!")
                break
        
        weights = rs / (rs.sum() + 1e-10)
    
    else:  # FedAvg
        weights = relative_shard_sizes / relative_shard_sizes.sum()
    
    for grad, weight in zip(gradients, weights):
        add_gradient_updates(aggregated_gradient, grad, weight=weight.item())
    
    for i in range(N + A):
        if use_sparsify and use_reputation:
            q_ratio = rs[i] / rs.max()
            reward_gradient = mask_grad_update_by_order(aggregated_gradient, q_ratio, 'layer')
        else:
            reward_gradient = aggregated_gradient
        
        if i < N:
            add_update_to_model(participant_models[i], reward_gradient)
        else:
            add_update_to_model(adv_models[i - N], reward_gradient)

    # Evaluation
    for i, model in enumerate(participant_models):
        loss, acc = evaluate(model, valid_loader, args['loss_fn'], device)
        valid_perfs[f'honest_{i}_loss'].append(loss.item())
        valid_perfs[f'honest_{i}_acc'].append(acc.item())
        
        if A > 0:
            _, target_acc, attack_success = evaluate(
                model, valid_loader, args['loss_fn'], device, label_flip='1-7'
            )
            adv_lf_perfs[f'honest_{i}_target_acc'].append(target_acc.item())
            adv_lf_perfs[f'honest_{i}_attack_success'].append(attack_success.item())
        
        loss, acc = evaluate(model, train_loaders[i], args['loss_fn'], device)
        local_perfs[f'honest_{i}_loss'].append(loss.item())
        local_perfs[f'honest_{i}_acc'].append(acc.item())

    if use_reputation:
        rs_dict.append(rs.detach().cpu().numpy())
        qs_dict.append((rs / rs.max()).detach().cpu().numpy())
        r_threshold.append(threshold * (1.0 / len(R_set)) if R_set else 0.0)

# Save Results
print("\nSaving results...")
folder = oj(
    'RFFL_results',
    args['dataset'],
    f"{args['split'][:3].upper()}_N{N}_A{A}_lf",
    f"r{int(use_reputation)}s{int(use_sparsify)}"
)
os.makedirs(folder, exist_ok=True)

if use_reputation:
    pd.DataFrame(np.array(rs_dict)).to_csv(oj(folder, 'rs.csv'), index=False)
    pd.DataFrame(np.array(qs_dict)).to_csv(oj(folder, 'qs.csv'), index=False)

pd.DataFrame(valid_perfs).to_csv(oj(folder, 'valid.csv'), index=False)
pd.DataFrame(local_perfs).to_csv(oj(folder, 'local.csv'), index=False)
if A > 0:
    pd.DataFrame(adv_lf_perfs).to_csv(oj(folder, 'attack_success.csv'), index=False)

with open(oj(folder, 'settings.txt'), 'w') as f:
    for k, v in args.items():
        f.write(f"{k}: {v}\n")
with open(oj(folder, 'settings.pkl'), 'wb') as f:
    pickle.dump(args, f)

torch.save({
    'server_model': server_model.state_dict(),
    'participant_models': [m.state_dict() for m in participant_models],
    'adv_models': [m.state_dict() for m in adv_models],
}, oj(folder, 'models.pth'))

print("Experiment completed successfully!")