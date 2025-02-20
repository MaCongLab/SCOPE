import os.path

import numpy as np
import torch
from tqdm import tqdm
import utils

import pandas as pd
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
import argparse


def generate_pdb_3d_graph(df_path,pdb_dir,graph_dir):
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    df = pd.read_csv(df_path)
    file_lst = os.listdir(pdb_dir)
    for i in tqdm(range(df.shape[0])):
        for file in file_lst:
            if file.startswith(f'{i}_unrelaxed_rank_001_alphafold2_ptm_model'):
                tmp_data = utils.get_graph_of_peptide_3d(os.path.join(pdb_dir,file),dist_th=3.5)
                torch.save(tmp_data,os.path.join(graph_dir,f'{i}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("df", help="path of the input csv file")
    parser.add_argument("pdb_dir", help="the path where the predicted structures are placed")
    parser.add_argument("graph_dir", help="the path where the generated 3D graphs are placed")
    args = parser.parse_args()
    generate_pdb_3d_graph(args.df,args.pdb_dir,args.graph_dir)