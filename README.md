# SCOPE

This repository is the official code of SCOPE: A 3D Structure-Based Peptide Cytotoxicity Prediction Framework

## Data preprocess
In SCOPE, the only preprocess required is predicting the 3D structure of peptides by Alphafold2 and preparing the 3D graphs.

If you are using our prepared 3D graphs, we have provided a zip file  `toxibtl_graphs.zip` in `/data` directory. You can simply unzip it.

If you are using your own data, we recommend you follow these steps:
1. Get your 3D structure of Peptides, we used the localized AF2，for AF2 localization please check https://github.com/kalininalab/alphafold_non_docker.

2. Please prepare a .csv file that has the formation like:
   
    | seq   | id |
    | ----------- | ----------- |
    | WWEGDCRTWDAPCNPAVECCFGVCRHRRCVLW      | 0       |
    | HLLQFNKMIKFETRKNAIPFYAFYGCYCGWGGRGRPKDATDRCCFVH   | 1        |

    Please ensure that the `id` column corresponds to the id of your predicted PDB file. If you use AF2, the corresponding PDB file should be named like `{id}_unrelaxed_rank_001_alphafold2_ptm_model`.

3. Run the command as follows:
   
   `python data_preprocess.py -df path_of_your_csv_file -pdb_dir path_of_PDB_dir -graph_dir path_of_output_dir`
   
    Then the 3D graph of the peptides will be generated.

## Model Training 
Run the command as follows:
`python train.py -exp_name SCOPE -data ../data/train.csv -save_folder ../save_model/`

## Model Testing
Run the command as follows:
`python val_test.py -data ../data/test.csv -save_model_path ../save_model/scope_model`





