import math

import rdkit.Chem
import torch
import os
import torch_geometric as pyg
import numpy as np
from rdkit import Chem
import ripser
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from rdkit.Chem import Descriptors,rdMolDescriptors,rdMolTransforms
# from chemprop.features import  features_generators
from Bio.PDB import DSSP
from Bio.PDB.PDBParser import PDBParser
from tqdm import tqdm

def getbond_type(bond):
    bond_type_enum = [Chem.BondType.SINGLE,
                      Chem.BondType.DOUBLE,
                      Chem.BondType.AROMATIC]
    return bond_type_enum.index(bond.GetBondType())


def getatom_type(atom):
    atom = atom.replace(' ','')
    atom_type_enum = ['S','O','C','N']
    return atom_type_enum.index(atom)


def gethybri_type(atom):
    atom_type_enum = [rdkit.Chem.HybridizationType.S,
                      rdkit.Chem.HybridizationType.SP,
                      rdkit.Chem.HybridizationType.SP2,
                      rdkit.Chem.HybridizationType.SP3]
    return atom_type_enum.index(atom.GetHybridization())

def getchiral_type(atom):
    chiral_type_enum = [rdkit.Chem.ChiralType.CHI_UNSPECIFIED,
                        rdkit.Chem.ChiralType.CHI_TETRAHEDRAL,
                        rdkit.Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                        rdkit.Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                        ]
    return chiral_type_enum.index(atom.GetChiralTag())


def get_graph_of_peptide(seq):
    patt_smarts = 'N-C=O'
    patt = Chem.MolFromSmarts(patt_smarts)

    mol = Chem.MolFromFASTA(seq)
    # Chem.Kekulize(mol, clearAromaticFlags=True)
    matches = mol.GetSubstructMatches(patt)
    node_idx_dict = {}
    peptide_key_num = len(matches)
    for matchid,match in enumerate(matches):
        for i in match:
            node_idx_dict[i] = matchid
    node_type_lst = [5]*peptide_key_num
    aromatic_type_lst = [0]*peptide_key_num
    hybri_type_lst = [4]*peptide_key_num
    edge_from = []
    edge_to = []
    edge_type = []

    for atom in mol.GetAtoms():
        if atom.GetIdx() not in node_idx_dict.keys():
            node_type_lst.append(getatom_type(atom.GetSymbol()))
            aromatic_type_lst.append(int(atom.GetIsAromatic()))
            hybri_type_lst.append(gethybri_type(atom))
            node_idx_dict[atom.GetIdx()]=len(node_type_lst)-1

    for bond in mol.GetBonds():
        a1 = node_idx_dict[bond.GetBeginAtom().GetIdx()]
        a2 = node_idx_dict[bond.GetEndAtom().GetIdx()]
        # if a1 == a2:
        #     continue
        edge_from.append(a1)
        edge_from.append(a2)
        edge_to.append(a2)
        edge_to.append(a1)
        btype = getbond_type(bond)
        edge_type.append(btype)
        edge_type.append(btype)

    # step = 20
    # for rnode in range(0,len(node_type_lst),int(step/2)):
    #     if (rnode+step)<len(node_type_lst):
    #         edge_from.append(rnode)
    #         edge_from.append(rnode+step)
    #         edge_to.append(rnode+step)
    #         edge_to.append(rnode)
    #         edge_type.append(3)
    #         edge_type.append(3)

    node_type = torch.tensor(node_type_lst, dtype=torch.long)
    hybri_type = torch.tensor(hybri_type_lst,dtype=torch.long)
    aromatic_type = torch.tensor(aromatic_type_lst,dtype=torch.long)
    edges = torch.tensor([edge_from, edge_to], dtype=torch.long)
    edge_type_lst = torch.tensor(edge_type, dtype=torch.long)
    data = Data(atom_type=node_type, edge_index=edges, edge_type=edge_type_lst,num_nodes=len(node_type_lst),aromatic_type = aromatic_type, hybri_type=hybri_type)
    return data

def get_graph_of_peptide_old(seq):
    mol = Chem.MolFromFASTA(seq)
    # Chem.Kekulize(mol, clearAromaticFlags=True)
    # global_node = []
    node_type_lst = []
    aromatic_type_lst = []
    formal_charge_lst = []
    numHs = []
    edge_from = []
    edge_to = []
    edge_type = []
    chi_type = []
    # edge_conj = []

    for atom in mol.GetAtoms():
        node_type_lst.append(getatom_type(atom.GetSymbol()))
        aromatic_type_lst.append(int(atom.GetIsAromatic()))
        formal_charge_lst.append(atom.GetFormalCharge())
        chi_type.append(getchiral_type(atom))
        numHs.append(atom.GetTotalNumHs())
        # global_node.append(0)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        edge_from.append(a1)
        edge_from.append(a2)
        edge_to.append(a2)
        edge_to.append(a1)
        btype = getbond_type(bond)
        edge_type.append(btype)
        edge_type.append(btype)

    node_type = torch.tensor(node_type_lst, dtype=torch.long)
    formal_charge = torch.tensor(formal_charge_lst,dtype=torch.float)
    numHs = torch.tensor(numHs,dtype=torch.long)
    aromatic_type = torch.tensor(aromatic_type_lst,dtype=torch.long)
    edges = torch.tensor([edge_from, edge_to], dtype=torch.long)
    edge_type_lst = torch.tensor(edge_type, dtype=torch.long)
    chi_type = torch.tensor(chi_type,dtype=torch.long)
    data = Data(atom_type=node_type, edge_index=edges, edge_type=edge_type_lst,num_nodes=len(node_type_lst),aromatic_type = aromatic_type, formal_charge=formal_charge,numHs=numHs,chi_type=chi_type)

    return data


def get_graph_of_peptide_bond_graph(seq):
    mol = Chem.MolFromFASTA(seq)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer(0)
    # Chem.Kekulize(mol, clearAromaticFlags=True)
    # global_node = []
    node_type_lst = []
    aromatic_type_lst = []
    formal_charge_lst = []
    edge_from = []
    edge_to = []
    edge_type = []
    # edge_conj = []

    for atom in mol.GetAtoms():
        node_type_lst.append(getatom_type(atom.GetSymbol()))
        aromatic_type_lst.append(int(atom.GetIsAromatic()))
        formal_charge_lst.append(atom.GetFormalCharge())
        # global_node.append(0)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        # if a1 == a2:
        #     continue
        edge_from.append(a1)
        edge_from.append(a2)
        edge_to.append(a2)
        edge_to.append(a1)
        btype = getbond_type(bond)
        edge_type.append(btype)
        edge_type.append(btype)

    node_type = torch.tensor(node_type_lst, dtype=torch.long)
    formal_charge = torch.tensor(formal_charge_lst,dtype=torch.float)
    aromatic_type = torch.tensor(aromatic_type_lst,dtype=torch.long)
    edges = torch.tensor([edge_from, edge_to], dtype=torch.long)
    edge_type_lst = torch.tensor(edge_type, dtype=torch.long)
    data = Data(atom_type=node_type, edge_index=edges, edge_type=edge_type_lst,num_nodes=len(node_type_lst),aromatic_type = aromatic_type, formal_charge=formal_charge)

    edge_node_type_lst = []
    angle_edge_from = []
    angle_edge_to = []
    bond_angles = []

    # rdMolTransforms.GetAngleDeg(conf, 0, 1, 2)
    for edge_idx in range(0, len(edge_from), 2):
        edge_node_type_lst.append(edge_type[edge_idx])
        for i in range(0,edge_idx,2):
            if (edge_from[i] in [edge_from[edge_idx],edge_to[edge_idx]]) or (edge_to[i] in [edge_from[edge_idx],edge_to[edge_idx]]):
                angle_edge_from.append(edge_idx//2)
                angle_edge_to.append(i//2)
                angle_edge_from.append(i//2)
                angle_edge_to.append(edge_idx//2)
                if edge_from[i] == edge_from[edge_idx]:
                    tmpangle = rdMolTransforms.GetAngleDeg(conf, edge_to[i], edge_from[i], edge_to[edge_idx])
                elif edge_from[i] == edge_to[edge_idx]:
                    tmpangle = rdMolTransforms.GetAngleDeg(conf, edge_to[i], edge_from[i], edge_from[edge_idx])
                elif edge_to[i] == edge_from[edge_idx]:
                    tmpangle = rdMolTransforms.GetAngleDeg(conf, edge_from[i], edge_to[i], edge_to[edge_idx])
                elif edge_to[i] == edge_to[edge_idx]:
                    tmpangle = rdMolTransforms.GetAngleDeg(conf, edge_from[i], edge_to[i], edge_from[edge_idx])

                bond_angles.append(tmpangle)
                bond_angles.append(tmpangle)

    edge_node_type = torch.tensor(edge_node_type_lst,dtype=torch.long)
    edge_graph_edges = torch.tensor([angle_edge_from, angle_edge_to], dtype=torch.long)
    bond_angles = torch.tensor(bond_angles,dtype=torch.float)
    edge_graph_data = Data(node_type=edge_node_type, edge_index=edge_graph_edges, edge_angles=bond_angles, num_nodes=len(edge_node_type))

    return data,edge_graph_data


def read_blosum62():
    aac_vocab = load_dict('../data/aac_vocab.txt')
    blosum62 = np.zeros((20,20))
    with open('../data/BLOSUM62.txt','r') as file:
        lines = file.readlines()[6:30]
        AAC_line = lines[0][3:].strip('\n')
        AACs =  AAC_line.split('  ')[:-1]
        for idx,line in enumerate(lines[1:]):
            line = line[3:-2].strip('\n').replace('  ',' ')
            nums = line.split(' ')[:-1]
            for num_idx,num in enumerate(nums):
                # if num=='':
                #     continue
                num = int(num)
                if AACs[num_idx] in aac_vocab and AACs[idx] in aac_vocab:
                    blosum62[aac_vocab[AACs[num_idx]],aac_vocab[AACs[idx]]]=num
    blosum62_ts = torch.tensor(blosum62,dtype=torch.long)
    torch.save(blosum62_ts,'../data/blosum62.pt')
    # print(blosum62)


def load_dict(file_path):
    new_dict = {}
    with open(file_path,'r') as dict_file:
        for line in dict_file.readlines():
            line = line.strip('\n')
            new_dict[line.split(',')[1]]=int(line.split(',')[0])
    return new_dict

def get_second_stru_type(symbol):
    sec_stry_enum = ['G','H','I','T','E','B','S','-']
    return sec_stry_enum.index(symbol)

def get_graph_of_peptide_3d_old(filename,dist_th):
    aac_vocab = load_dict('../data/aac_vocab.txt')
    p = PDBParser(PERMISSIVE=1)
    structure_id = "tmp"
    s = p.get_structure(structure_id, filename)
    model = s[0]
    rd = DSSP(model, filename, dssp='mkdssp')
    rd = rd.property_list
    chain = model['A']
    node_type_lst = []
    sec_stru_lst = []
    atom_res_lst = []
    edge_from = []
    edge_to = []
    edge_type = []
    pos_mat = []
    for res_id,res in enumerate(chain):
        tmp_second_stru_type = get_second_stru_type(rd[res_id][2])
        tmp_res_name = aac_vocab[rd[res_id][1]]
        # print(res.get_resname())
        for atom in res.get_list():
            node_type_lst.append(getatom_type(atom.get_fullname()[:2]))
            atom_res_lst.append(tmp_res_name)
            sec_stru_lst.append(tmp_second_stru_type)
            pos_mat.append(atom.get_coord())
    pos_mat = np.vstack(pos_mat)

    for i in range(len(node_type_lst)):
        for j in range(i):
            if i == j:
                continue
            tmp_dist = np.sqrt(np.sum((pos_mat[i]-pos_mat[j])**2))
            if tmp_dist<=1.7:
                edge_from.append(i)
                edge_from.append(j)
                edge_to.append(j)
                edge_to.append(i)
                edge_type.append(0)
                edge_type.append(0)


    node_type = torch.tensor(node_type_lst, dtype=torch.long)
    atom_res_type = torch.tensor(atom_res_lst,dtype=torch.long)
    sec_stru_type = torch.tensor(sec_stru_lst,dtype=torch.long)
    edges = torch.tensor([edge_from, edge_to], dtype=torch.long)
    edge_type_lst = torch.tensor(edge_type, dtype=torch.long)
    data = Data(atom_type=node_type, sec_stru_type=sec_stru_type, atom_res_type=atom_res_type,edge_index=edges, edge_type=edge_type_lst,num_nodes=len(node_type_lst))
    # data = torch.save(data,'')
    # data = torch.load('tmp.pt')
    # print(data)
    return data

def get_graph_of_peptide_3d(filename,dist_th):
    aac_vocab = load_dict('../data/aac_vocab.txt')
    p = PDBParser(PERMISSIVE=1)
    structure_id = "tmp"
    s = p.get_structure(structure_id, filename)
    model = s[0]
    rd = DSSP(model, filename, dssp='mkdssp')
    rd = rd.property_list
    chain = model['A']
    node_type_lst = []
    sec_stru_lst = []
    atom_res_lst = []
    edge_from = []
    edge_to = []
    edge_type = []
    pos_mat = []
    for res_id,res in enumerate(chain):
        tmp_second_stru_type = get_second_stru_type(rd[res_id][2])
        tmp_res_name = aac_vocab[rd[res_id][1]]
        # print(res.get_resname())
        for atom in res.get_list():
            node_type_lst.append(getatom_type(atom.get_fullname()[:2]))
            atom_res_lst.append(tmp_res_name)
            sec_stru_lst.append(tmp_second_stru_type)
            pos_mat.append(atom.get_coord())
    pos_mat = np.vstack(pos_mat)

    for i in range(len(node_type_lst)):
        for j in range(i):
            if i == j:
                continue
            tmp_dist = np.sqrt(np.sum((pos_mat[i]-pos_mat[j])**2))
            if tmp_dist<=3.5:
                edge_from.append(i)
                edge_from.append(j)
                edge_to.append(j)
                edge_to.append(i)
                edge_type.append(0)
                edge_type.append(0)


    node_type = torch.tensor(node_type_lst, dtype=torch.long)
    atom_res_type = torch.tensor(atom_res_lst,dtype=torch.long)
    sec_stru_type = torch.tensor(sec_stru_lst,dtype=torch.long)
    edges = torch.tensor([edge_from, edge_to], dtype=torch.long)
    edge_type_lst = torch.tensor(edge_type, dtype=torch.long)
    data = Data(atom_type=node_type, sec_stru_type=sec_stru_type, atom_res_type=atom_res_type,edge_index=edges, edge_type=edge_type_lst,num_nodes=len(node_type_lst))
    # data = torch.save(data,'')
    # data = torch.load('tmp.pt')
    # print(data)
    return data

def get_graph_of_peptide_3dresi(filename):
    aac_vocab = load_dict('../data/aac_vocab.txt')
    p = PDBParser(PERMISSIVE=1)
    structure_id = "tmp"
    s = p.get_structure(structure_id, filename)
    model = s[0]
    rd = DSSP(model, filename, dssp='mkdssp')
    rd = rd.property_list
    chain = model['A']
    node_type_lst = []
    sec_stru_lst = []
    atom_res_dict = {}
    atom_lst = []
    edge_from = []
    edge_to = []
    edge_type = []
    pos_mat = []
    for res_id, res in enumerate(chain):
        tmp_second_stru_type = get_second_stru_type(rd[res_id][2])
        tmp_res_name = aac_vocab[rd[res_id][1]]
        node_type_lst.append(tmp_res_name+4)
        res_node_id = len(node_type_lst)-1
        sec_stru_lst.append(tmp_second_stru_type)
        atom_res_dict[res_node_id]=[]

        for atom in res.get_list():
            node_type_lst.append(getatom_type(atom.get_fullname()[:2]))
            atom_lst.append(len(node_type_lst)-1)
            atom_res_dict[res_node_id].append(len(node_type_lst)-1)
            sec_stru_lst.append(tmp_second_stru_type)
            pos_mat.append(atom.get_coord())
    pos_mat = np.vstack(pos_mat)

    for i in range(len(atom_lst)):
        for j in range(i):
            if i == j:
                continue
            tmp_dist = np.sqrt(np.sum((pos_mat[i] - pos_mat[j]) ** 2))
            if tmp_dist <= 3.5:
                edge_from.append(atom_lst[i])
                edge_from.append(atom_lst[j])
                edge_to.append(atom_lst[j])
                edge_to.append(atom_lst[i])
                edge_type.append(0)
                edge_type.append(0)

    for i in atom_res_dict.keys():
        for j in atom_res_dict[i]:
            edge_from.append(i)
            edge_from.append(j)
            edge_to.append(j)
            edge_to.append(i)
            edge_type.append(1)
            edge_type.append(1)

    last_resi_node = None
    for i in range(len(node_type_lst)):
        if i in atom_res_dict:
            if last_resi_node is not None:
                edge_from.append(i)
                edge_from.append(last_resi_node)
                edge_to.append(last_resi_node)
                edge_to.append(i)
                edge_type.append(2)
                edge_type.append(2)
            last_resi_node = i


    node_type = torch.tensor(node_type_lst, dtype=torch.long)
    sec_stru_type = torch.tensor(sec_stru_lst, dtype=torch.long)
    edges = torch.tensor([edge_from, edge_to], dtype=torch.long)
    edge_type_lst = torch.tensor(edge_type, dtype=torch.long)
    data = Data(atom_type=node_type, sec_stru_type=sec_stru_type, edge_index=edges,
                edge_type=edge_type_lst, num_nodes=len(node_type_lst))
    return data

def get_ph_seqs(file_name):
    p = PDBParser(PERMISSIVE=1)
    s = p.get_structure('tmp', file_name)
    model = s[0]
    chain = model['A']
    pos_mat = []
    for res in chain:
        for atom in res.get_list():
            pos_mat.append(atom.get_coord())
    pos_mat = np.vstack(pos_mat)
    # Generate data on customer purchases
    diagrams = ripser.ripser(pos_mat)['dgms']
    h0 = diagrams[0][:-1,1]
    h0 = torch.tensor(h0,dtype=torch.float)
    h0 = torch.unsqueeze(h0,dim=-1)
    print(h0)
    return h0

def TP(pred,label):
    count = 0
    for i in range(len(pred)):
        if (pred[i]==label[i] and pred[i]==1):
            count+=1
    return count

def FP(pred,label):
    count = 0
    for i in range(len(pred)):
        if (pred[i]!=label[i] and pred[i]==1):
            count+=1
    return count

def TN(pred,label):
    count = 0
    for i in range(len(pred)):
        if (pred[i]==label[i] and pred[i]==0):
            count+=1
    return count

def FN(pred,label):
    count = 0
    for i in range(len(pred)):
        if (pred[i]!=label[i] and pred[i]==0):
            count+=1
    return count


def SN(pred,label):
    tp = TP(pred,label)
    fn = FN(pred,label)
    return tp/(tp+fn)

def SP(pred,label):
    tn = TN(pred,label)
    fp = FP(pred,label)
    return tn/(tn+fp)

def FDR(pred,label):
    tp = TP(pred,label)
    fp = FP(pred,label)
    return fp/(tp+fp)

def ACC(pred,label):
    tp = TP(pred,label)
    tn = TN(pred, label)
    fp = FP(pred,label)
    fn = FN(pred,label)
    return (tp+tn)/(fp+fn+tp+tn)

def MCC(pred,label):
    tp = TP(pred, label)
    tn = TN(pred, label)
    fp = FP(pred, label)
    fn = FN(pred, label)
    return (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))