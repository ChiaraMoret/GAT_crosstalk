# from network_experiments import *
from GAT_BCE_models import *
import os
import numpy as np
import random
from MiGe_dataset import *

def set_seed(seed):

    # import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)       
        torch.cuda.manual_seed_all(seed)   
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
def pos_pt(split_set):
    all_labels = np.array([data.y for data in split_set])
    pos_split_pt = all_labels.sum() * 100 / len(all_labels)
    return pos_split_pt
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparametrs
train_batch_size = 1
val_batch_size = 1
hidden_channels = 64
num_layers = 3
n_heads = 5
dropout = 0.0
l1_lambda = 0.0
wd = 0.0

study = "Kidney renal clear cell carcinoma"
tissue = study.split()[0]
epochs = 3
lr = 0.00005
make_undirected = True
save_model=True
save_results=True

dataset_tissue = load_mirna_dataset(status='all', studies=[study], make_undirected=make_undirected)
full_loader = DataLoader(dataset_tissue, batch_size=1, shuffle=False)

seeds = [42]
for seed in seeds:
    set_seed(seed)
    g=torch.Generator()
    g.manual_seed(seed)
    
    train_set, val_set, test_set = split_dataset(
    dataset_tissue, generator=g, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    pos_train_pt = pos_pt(train_set)
    pos_test_pt = pos_pt(test_set)
    pos_val_pt = pos_pt(val_set)
    
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, generator=g)
    
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, generator=g)
    
    model = GAT_BCE(in_channels=1, hidden_channels=hidden_channels,
                    num_layers=num_layers, heads=n_heads, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    pos_number = sum(dataset_tissue.labels)
    neg_number = len(dataset_tissue.labels) - pos_number
    pos_weight = neg_number / pos_number
    print(f'pos_weight is {pos_weight:.4f}')
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    
    all_train_losses, all_val_losses, all_confusions = [], [], []
    all_f1_1, all_precision_1, all_recall_1 = [], [], []
    all_f1_0, all_precision_0, all_recall_0 = [], [], []
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device, l1_lambda=l1_lambda)
        val_loss = evaluate(model, val_loader, criterion, device)
        f1_1, precision_1, recall_1, cm = test(model, test_loader, device)
        tn, fp, fn, tp = cm.ravel().tolist()
        
        epsilon = 1e-10 
    
        precision_0 = tn / (tn + fn + epsilon)
        recall_0 = tn / (tn + fp + epsilon)
        f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + epsilon)
        
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f},"# Val Loss: {val_loss:.4f}, "
            f"F1: {f1_1:.4f}, Precision: {precision_1:.4f}, Recall: {recall_1:.4f}, "
            f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}"
        )
    
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        
        all_confusions.append([tn, fp, fn, tp])
    
        all_f1_1.append(f1_1)
        all_precision_1.append(precision_1)
        all_recall_1.append(recall_1)
    
        all_f1_0.append(f1_0)
        all_precision_0.append(precision_0)
        all_recall_0.append(recall_0)
    
    results = {
        "train_pos": pos_train_pt,
        "val_pos": pos_val_pt,
        "test_pos": pos_test_pt,
        "train_loss": torch.tensor(all_train_losses),
        "val_loss": torch.tensor(all_val_losses),
        "F1, c1": all_f1_1, 
        "Precision, c1": all_precision_1, 
        "Recall, c1": all_recall_1, 
        "F1, c0": all_f1_0, 
        "Precision, c0": all_precision_0, 
        "Recall, c0": all_recall_0, 
        "Confusion Matrix": all_confusions,
        "hparams": {
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "n_heads": n_heads,
            "dropout": dropout,
            "lr": lr,
            "epochs": epochs,
            "make_undirected": make_undirected,
            "l1": l1_lambda,
            "wd": wd
        }
    }

    results_folder = 'results'
   
    if save_results:
        results_parts = {
        "study": study.replace(" ", "_"),
        "epochs": epochs,
        "trainbs": train_batch_size,
        "valbs": val_batch_size,
        "hd": hidden_channels,
        "layers": num_layers,
        "heads": n_heads,
        "do": dropout,
        "lr": lr,
        "undirected": make_undirected,
        "fixseed": seed,
        "l1": l1_lambda,
        "wd": wd
        }
        
        results_tag = "_".join(f"{k}{v}" for k, v in results_parts.items())

        filename = f"{results_folder}/results_{tag}.pt"
        
        torch.save(results, filename)
        print(f"Results saved to {filename}")

    
    if save_model:
        model_parts = {
        "tissue": f'{tissue}_model',
        "epochs": epochs,
        "trainbs": train_batch_size,
        "valbs": val_batch_size,
        "hd": hidden_channels,
        "layers": num_layers,
        "heads": n_heads,
        "do": dropout,
        "lr": lr,
        "undirected": make_undirected,
        "fixseed": seed,
        "l1": l1_lambda,
        "wd": wd
        }
        
        model_tag = "_".join(f"{k}{v}" for k, v in model_parts.items())
        # # saving actual model
        saving_folder = f'{results_folder}/{tissue}_model_{seed}'
        os.makedirs(saving_folder, exist_ok=True)
        model_path = f'{saving_folder}/{model_tag}.pth'
        
        torch.save(model, model_path)
        trainset_path = f'{saving_folder}/train_set.pt'
        
        torch.save(train_set, trainset_path)
        
        testset_path = f'{saving_folder}/test_set.pt'
        torch.save(test_set, testset_path)
        
        valset_path = f'{saving_folder}/val_set.pt'
        torch.save(val_set, valset_path)
        print(f'Model saved to {model_path}')