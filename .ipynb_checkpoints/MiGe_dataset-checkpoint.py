import os ##added
import pandas as pd
from GAT_dataset import *


def filter_expression_df(expression_df, patients, studies):

        candidate_codes = patients.loc[
            patients['study'].isin(studies), 'sample_code'
        ]
        # Subset expression_df columns by candidate_codes
        expression_df = expression_df.loc[:, expression_df.columns.isin(candidate_codes)]
        return expression_df


def load_mirna_dataset(status='all', studies=None, tissue_labels=True, status_labels=True, make_undirected=False):

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # mirtarbase
    mirtarbase_path = os.path.join(BASE_DIR, "data", "MirTarBase", "miRTarBase_MTI.csv")
    mirtarbase_df = pd.read_csv(mirtarbase_path)
    
    mirtarbase_df = mirtarbase_df[mirtarbase_df['Species (miRNA)'] == 'hsa']
    mirtarbase_df['miRNA'] = mirtarbase_df['miRNA'].str.lower()
    
    patients_path = os.path.join(BASE_DIR, "data", "TCGA", "patients.csv")
    patients =  pd.read_csv(patients_path)
    pat_codes = list(patients.sample_code)
    
    
    #### correct import and filtering block
    mirna_path = os.path.join(BASE_DIR, "data", "TCGA", "miRNA_expression.xena.gz")
    mirna_df = pd.read_csv(mirna_path, compression="gzip", sep="\t")
    mirna_df['sample'] = mirna_df['sample'].str.lower()
    mirna_df = mirna_df[['sample'] + pat_codes] # keep only patients in our subset
    
    mrna_path = os.path.join(BASE_DIR, "data", "TCGA", "mRNA_expression.xena.gz")
    mrna_df = pd.read_csv(mrna_path, compression="gzip", sep="\t")
    mrna_df = mrna_df.drop_duplicates(subset='sample', keep=False)
    mrna_df = mrna_df[['sample'] + pat_codes] # keep only patients in our subset
    # mrna_df = mrna_df.dropna()
    
    
    ## now filtering based on mirtarbase connections
    mirna_df= mirna_df[mirna_df['sample'].isin(mirtarbase_df['miRNA'].unique())]
    mirtarbase_df = mirtarbase_df[mirtarbase_df['miRNA'].isin(mirna_df['sample'].unique())]
    
    mrna_df = mrna_df[mrna_df['sample'].isin(mirtarbase_df['Target Gene'].unique())]
    mirtarbase_df = mirtarbase_df[mirtarbase_df['Target Gene'].isin(mrna_df['sample'].unique())]
    mrna_df = mrna_df.fillna(0.0)
    
    mirna_df = mirna_df.set_index('sample')
    mrna_df = mrna_df.set_index('sample')

    ## cleanup
    expression_df = pd.concat((mirna_df,mrna_df), axis=0)
    
    if studies is not None:
        if status!='all':
            patients = patients[patients['status'] == status]
            
        expression_df = filter_expression_df(expression_df, patients, studies)

    
    ints_df = mirtarbase_df[['miRNA','Target Gene']].drop_duplicates()
    
    dataset = GATDataset(expression_df, ints_df, col1='miRNA', col2='Target Gene', tissue_labels=tissue_labels, status_labels=status_labels, patients=patients, make_undirected=make_undirected)
    print('loaded mirna dataset')
    print(len(dataset))
    
    return dataset
