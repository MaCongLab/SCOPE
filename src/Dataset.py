import torch
import pandas as pd
import numpy as np
from utils import get_graph_of_peptide_old

class protein_dataset(torch.utils.data.Dataset):
    def __init__(self,data_path,maxlen,pos_dim):
        super().__init__()
        self.df = pd.read_csv(data_path)
        self.aac_vocab = self.load_dict('../data/aac_vocab.txt')
        self.aac2_vocab = self.load_dict('../data/aac2_vocab.txt')
        self.length = self.df.shape[0]
        self.maxlen = maxlen
        self.pos_dim =  pos_dim
        self.blosum = torch.load('../data/blosum62.pt',map_location='cpu').type(torch.float)
        self.aaindex2 = torch.load('../data/aac_index2.pt',map_location='cpu').type(torch.float)


    def __len__(self):
        return self.length

    def load_dict(self,file_path):
        new_dict = {}
        with open(file_path,'r') as dict_file:
            for line in dict_file.readlines():
                line = line.strip('\n')
                new_dict[line.split(',')[1]]=int(line.split(',')[0])
        return new_dict

    def _get_sinusoid_encoding_table(self,n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(self.maxlen, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def __getitem__(self, idx):
        tmp_seq = self.df.loc[idx,'seq']
        tmp_id = self.df.loc[idx,'id']
        tmp_label = self.df.loc[idx,'label']
        # print(tmp_id)
        tmp_seq = tmp_seq.upper()

        tmp_aac2_prop = torch.zeros(400, dtype=torch.float)
        for i in range(1,len(tmp_seq)):
            tmp_aac2_prop[self.aac2_vocab[tmp_seq[i-1:i+1]]] += 1

        tmp_aac2_prop /= len(tmp_seq)-1

        tmp_graph = get_graph_of_peptide_old(tmp_seq)
        tmp_graph_3d = torch.load(f'../data/toxibtl_graphs/{tmp_id}.pt')
        tmp_seq = list(map(lambda x:self.aac_vocab[x],tmp_seq))
        seq_len = len(tmp_seq)


        tmp_aac_prop = torch.zeros(20,dtype=torch.float)
        for aac in tmp_seq:
            tmp_aac_prop[aac]+=1
        tmp_aac_prop /= seq_len

        aac_pair = torch.zeros((94,self.maxlen,self.maxlen))
        for i in range(len(tmp_seq)):
            for j in range(len(tmp_seq)):
                aac_pair[0,i,j] = self.blosum[tmp_seq[i],tmp_seq[j]]
                aac_pair[1:,i,j] = self.aaindex2[:,tmp_seq[i],tmp_seq[j]]
        aac_pair = torch.unsqueeze(aac_pair,dim=0)



        seq_len_t = torch.tensor(seq_len,dtype=torch.long)

        padding_mask = [False] * (len(tmp_seq))+[True]*(self.maxlen-len(tmp_seq))
        padding_mask = torch.BoolTensor(padding_mask)
        tmp_seq = tmp_seq+[20]*(self.maxlen-len(tmp_seq))
        tmp_seq = torch.tensor(tmp_seq,dtype=torch.long)
        tmp_label = tmp_label

        return tmp_seq,padding_mask,seq_len_t,tmp_graph,tmp_graph_3d,tmp_aac_prop,tmp_aac2_prop,aac_pair,tmp_label
