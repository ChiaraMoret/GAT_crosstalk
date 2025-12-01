import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch.utils.data import random_split


class GATDataset(Dataset):
    def __init__(self, df_data, df_edges, col1='node1', col2='node2', tissue_labels=True, status_labels=True, patients=None, make_undirected=False):
        """
        labels here represent a labeling for the columns which can be used for splitting 
        """
        super().__init__()
        self.df_data = df_data
        self.df_edges = df_edges
        self.patients = patients
        self.labels_dict, self.labels = self.get_labels(tissue_labels, status_labels) 

        # Map node names to integer indices
        self.node_to_idx = {node: i for i, node in enumerate(df_data.index)}
        self.idx_to_node = {idx : node for node, idx in self.node_to_idx.items()}
        
        # Prepare edge_index once (shared by all samples)
        edges = df_edges[[col1, col2]].apply(lambda col: col.map(self.node_to_idx.get)).values.T
        
        self.edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
        
        if make_undirected:
            self.edge_index = to_undirected(self.edge_index)
            
    def len(self):
        # Number of graph samples = number of columns in df_data
        return self.df_data.shape[1]

    def get(self, idx):
        # For sample idx, get node features for all nodes (one feature per node)
        x = torch.tensor(self.df_data.iloc[:, idx].values, dtype=torch.float).unsqueeze(1)  # shape [num_nodes, 1]
        
        # Create Data object for PyG
        data = Data(x=x, edge_index=self.edge_index, y= self.labels[idx])
        data.name = self.df_data.columns[idx]
        data.mask = torch.ones(x.size(0), dtype=torch.bool)  # mask all nodes by default

        return data

    def get_labels(self, tissue_labels, status_labels):
        if tissue_labels or status_labels:
            assert self.patients is not None, "Patient list must be provided if labels required"

            self.patients['label'] = (
                (self.patients['study'] if tissue_labels else '') + ' ' +
                (self.patients['status'] if status_labels else '')
            )
            labels = self.patients.set_index('sample_code').loc[self.df_data.columns, 'label']
            labels = labels.str.strip().tolist()  # ordered list
            label_to_int = {label: i for i, label in enumerate(dict.fromkeys(labels))}
            int_labels = torch.tensor([label_to_int[label] for label in labels])
            return label_to_int, int_labels
        return None, None




# logic to handle splitting
def section_dataset(dataset):
    assert dataset.labels is not None, "Dataset must have class labels"

    for cls in dataset.labels.unique():
        return None
        
def merge_subsets(subset_list, deduplicate=False, sort_indices=True):
    if not subset_list:
        raise ValueError("The subset list is empty.")
    
    base_dataset = subset_list[0].dataset
    for subset in subset_list:
        if subset.dataset is not base_dataset:
            raise ValueError("All subsets must come from the same dataset.")

    all_indices = []
    for subset in subset_list:
        all_indices.extend(subset.indices)

    if deduplicate:
        all_indices = list(set(all_indices))

    if sort_indices:
        all_indices = sorted(all_indices)

    return Subset(base_dataset, all_indices)
    
def split_dataset(dataset, generator, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    total = len(dataset)
    lengths = [
        int(total * train_ratio),
        int(total * val_ratio),
        total - int(total * train_ratio) - int(total * val_ratio)
    ]
    return random_split(dataset, lengths, generator=generator)

def split_dataset_balanced(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    sectioned_datasets = section_dataset(dataset)
    train_dataset, val_dataset, test_dataset = [], [], []
    for ds in sectioned_datasets:
        ds_train, ds_val, ds_test = split_dataset(ds, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        train_dataset.append(ds_train)
        val_dataset.append(ds_val)
        test_dataset.append(ds_test)

    return merge_subsets(train_dataset), merge_subsets(val_dataset), merge_subsets(test_dataset), 

## stratified 
# def split_dataset(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
#     assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

#     # Extract subtype labels from the dataset
#     labels = [get_subtype_label(dataset[i]) for i in range(len(dataset))]

#     sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=seed)
#     train_idx, temp_idx = next(sss1.split(range(len(dataset)), labels))

#     # Split temp set into val and test
#     temp_labels = [labels[i] for i in temp_idx]
#     val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
#     sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_ratio_adjusted), random_state=seed)
#     val_idx, test_idx = next(sss2.split(range(len(temp_idx)), temp_labels))
#     val_idx = [temp_idx[i] for i in val_idx]
#     test_idx = [temp_idx[i] for i in test_idx]

#     return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
