import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import aggr
from MGAT import MGAT
import numpy as np

class model(torch.nn.Module):

    def __init__(self,feat_dim,trans_layer_num):
        super().__init__()
        self.activation = nn.ELU(alpha=0.5)
        device = 'cuda'
        aac_index = torch.load('../data/aac_index.pt',map_location='cpu').type(torch.float)
        aac_index = torch.concat([aac_index,torch.zeros((1,566))],dim=0)
        self.aac_index_emb = nn.Embedding(21,padding_idx=20,embedding_dim=566).from_pretrained(aac_index,freeze=True)
        self.aac_emb = nn.Embedding(21,padding_idx=20,embedding_dim=feat_dim)
        self.blosum62 = torch.load('../data/blosum62.pt',map_location='cpu').type(torch.float)
        self.blosum62 = torch.concat([self.blosum62,torch.zeros((1,20))],dim=0)
        self.blosum62_emb = nn.Embedding(21,padding_idx=20,embedding_dim=20).from_pretrained(self.blosum62,freeze=True)
        self.blosum_norm = nn.LayerNorm(20)
        self.trans_layer_num = trans_layer_num
        self.pos_encode = self._get_sinusoid_encoding_table(50,feat_dim)
        self.p_dim = feat_dim

        self.seq_1pcnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2*feat_dim, kernel_size=(1,feat_dim+566+20)),
            nn.BatchNorm2d(2*feat_dim),
            self.activation,
            nn.Conv2d(in_channels=2*feat_dim, out_channels=feat_dim, kernel_size=(1,1)),
            nn.BatchNorm2d(feat_dim),
            self.activation,
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(feat_dim),
            self.activation,
        )

        self.seq_2pcnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2,feat_dim)),
            nn.BatchNorm2d(32),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            self.activation,
            nn.Conv2d(in_channels=64, out_channels=feat_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(feat_dim),
            self.activation,
        )

        self.pair_cnn = nn.Sequential(
            nn.Conv2d(in_channels=94+128, out_channels=128, kernel_size=(1,1)),
            self.activation,
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1),
            self.activation,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
            self.activation,
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            self.activation,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            self.activation,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1)),
            self.activation,
            nn.MaxPool2d((2,2)),
        )

        self.pair_ln = nn.Sequential(
            nn.Linear(12*12*256,1024),
            nn.BatchNorm1d(1024),
            self.activation,
            nn.Linear(1024, feat_dim),
            nn.BatchNorm1d(feat_dim),
            self.activation
        )

        self.lstm1 = nn.LSTM(input_size=self.p_dim,hidden_size=feat_dim,num_layers=3,bidirectional=True,batch_first=True)
        self.lstm2 = nn.LSTM(input_size=feat_dim, hidden_size=feat_dim, num_layers=3, bidirectional=True,
                             batch_first=True)

        self.lstm1_ln = nn.Sequential(
            nn.Linear(4*feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            self.activation,
        )
        self.lstm2_ln = nn.Sequential(
            nn.Linear(4 * feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            self.activation,
        )

        self.agg1 = aggr.MeanAggregation()
        self.transformer_layers = nn.ModuleList()
        for i in range(self.trans_layer_num):
            self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=self.p_dim, nhead=4, dim_feedforward=4*feat_dim,
                                                       batch_first=True,activation=self.activation,norm_first=False))

        self.atom_embedding = nn.Embedding(num_embeddings=6, embedding_dim=feat_dim)
        self.atom_aromatic_embedding = nn.Embedding(num_embeddings=2, embedding_dim=feat_dim)
        self.num_Hs_embedding = nn.Embedding(num_embeddings=10,embedding_dim=64)
        self.bond_embedding = nn.Embedding(num_embeddings=3, embedding_dim=feat_dim)

        self.gat2d_atom_ln = nn.Sequential(
            nn.Linear(feat_dim+feat_dim+1+64,feat_dim),
            self.activation
        )
        self.gatlayer1 = MGAT(in_channels=[feat_dim,feat_dim],out_channels=feat_dim,edge_dim=feat_dim)
        self.gatlayer2 = MGAT(in_channels=[feat_dim,feat_dim], out_channels=feat_dim, edge_dim=feat_dim)
        self.gatlayer3 = MGAT(in_channels=[feat_dim,feat_dim],out_channels=feat_dim, edge_dim=feat_dim)
        self.gatlayer4 = MGAT(in_channels=[feat_dim, feat_dim], out_channels=feat_dim, edge_dim=feat_dim)
        self.gatlayer5 = MGAT(in_channels=[feat_dim, feat_dim], out_channels=feat_dim, edge_dim=feat_dim)
        self.gat2d_ln =  self.gat_ln = nn.Sequential(
                            nn.Linear(5*feat_dim, feat_dim),
                            self.activation
                        )

        self.atom3d_embedding = nn.Embedding(num_embeddings=4, embedding_dim=feat_dim)
        self.bond3d_embedding = nn.Embedding(num_embeddings=3, embedding_dim=feat_dim)
        self.sec_stru_embedding = nn.Embedding(8,embedding_dim=64)
        self.gat3dlayer1 = MGAT(in_channels=[feat_dim+64,feat_dim+64], out_channels=feat_dim, edge_dim=feat_dim,heads=2)
        self.gat3dlayer2 = MGAT(in_channels=[2*feat_dim,2*feat_dim], out_channels=feat_dim, edge_dim=feat_dim,heads=2)
        self.gat3dlayer3 = MGAT(in_channels=[2*feat_dim,2*feat_dim], out_channels=feat_dim, edge_dim=feat_dim,heads=2)
        self.gat3dlayer4 = MGAT(in_channels=[2 * feat_dim,2*feat_dim], out_channels=feat_dim, edge_dim=feat_dim, heads=2)




        self.gat3d_ln = nn.Sequential(
            nn.Linear(8*feat_dim, 4*feat_dim),
            nn.Dropout(0.2),
            self.activation,
            nn.Linear(4 * feat_dim, 2*feat_dim),
            nn.Dropout(0.2),
            self.activation,
            nn.Linear(2 * feat_dim, feat_dim),
            nn.Dropout(0.2),
            self.activation,
        )

        self.dpc_ln = nn.Sequential(
            nn.Linear(420,2*feat_dim),
            nn.BatchNorm1d(2*feat_dim),
            self.activation,
            nn.Linear(2*feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            self.activation
        )

        self.trans_out_ln = nn.Sequential(
            nn.Linear(feat_dim,feat_dim),
            nn.BatchNorm1d(feat_dim),
            self.activation,
            nn.Linear(feat_dim, 2),
        )
        self.lstm_out_ln = nn.Sequential(
            nn.Linear(2*feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            self.activation,
            nn.Linear(feat_dim, 2),
        )
        self.graph2d_out_ln = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            self.activation,
            nn.Linear(feat_dim, 2),
        )
        self.graph3d_out_ln = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            self.activation,
            nn.Linear(feat_dim, 2),
        )
        self.out_ln = nn.Sequential(
            nn.Linear(7*feat_dim,2*feat_dim),
            nn.BatchNorm1d(2*feat_dim),
            self.activation,
            nn.Linear(2 * feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            self.activation,
            nn.Linear(feat_dim, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Linear(64, 2)
        )

    def _get_sinusoid_encoding_table(self,n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0).cuda()

    def forward(self,seq_batch,pad_mask_batch,batch_seq_len,batch_graph,batch_3d_graph,batch_aac_prop,batch_aac2_prop,batch_aac_pair):
        aac_index_emb = self.aac_index_emb(seq_batch)
        blosum_emb = self.blosum62_emb(seq_batch)
        blosum_emb = self.blosum_norm(blosum_emb)
        seq_emb = self.aac_emb(seq_batch)
        seq_emb = torch.concat([seq_emb,aac_index_emb,blosum_emb], dim=-1)
        seq_emb = torch.unsqueeze(seq_emb,dim=1)
        seq_emb = self.seq_1pcnn(seq_emb)
        seq_emb = torch.squeeze(seq_emb,dim=-1)
        seq_emb = torch.permute(seq_emb,[0,2,1])
        seq_emb[seq_batch == 20, :] = 0
        trans_emb = seq_emb + self.pos_encode
        seq_emb_2p = torch.unsqueeze(seq_emb, dim=1)

        seq_emb_pair_tmp1 = torch.unsqueeze(seq_emb,dim=-1)
        seq_emb_pair_tmp1 = torch.permute(seq_emb_pair_tmp1, dims=[0,2,1,3])
        seq_emb_pair_tmp2 = torch.permute(seq_emb_pair_tmp1, dims=[0,1,3,2])
        seq_emb_pair = torch.matmul(seq_emb_pair_tmp1, seq_emb_pair_tmp2)
        seq_emb_pair = seq_emb_pair / math.sqrt(128)
        batch_aac_pair = torch.concat([batch_aac_pair, seq_emb_pair], dim=1)
        batch_aac_pair = self.pair_cnn(batch_aac_pair)
        batch_aac_pair = torch.flatten(batch_aac_pair, start_dim=1)
        batch_aac_pair = self.pair_ln(batch_aac_pair)





        seq_emb_2p = self.seq_2pcnn(seq_emb_2p)
        seq_emb_2p = torch.squeeze(seq_emb_2p, dim=-1)
        seq_emb_2p = torch.permute(seq_emb_2p, [0, 2, 1])




        atom_emb = self.atom_embedding(batch_graph['atom_type'])
        atom_aromatic_emb = self.atom_aromatic_embedding(batch_graph['aromatic_type'])
        atom_numhs = self.num_Hs_embedding(batch_graph['numHs'])
        bond_emb = self.bond_embedding(batch_graph['edge_type'])
        formal_charge = torch.unsqueeze(batch_graph['formal_charge'],dim=-1)
        atom_emb = torch.concat([atom_emb,atom_aromatic_emb,atom_numhs,formal_charge],dim=-1)
        atom_emb = self.gat2d_atom_ln(atom_emb)
        atom_emb1,d2bond_emb1 = self.gatlayer1(atom_emb,batch_graph.edge_index,edge_attr=bond_emb)
        atom_emb1 = self.activation(atom_emb1)
        atom_emb2,d2bond_emb2 = self.gatlayer2(atom_emb1, batch_graph.edge_index, edge_attr=d2bond_emb1)
        atom_emb2 = self.activation(atom_emb2)
        atom_emb3,d2bond_emb3 = self.gatlayer3(atom_emb2, batch_graph.edge_index, edge_attr=d2bond_emb2)
        atom_emb3 = self.activation(atom_emb3)
        atom_emb4,d2bond_emb4 = self.gatlayer4(atom_emb3, batch_graph.edge_index, edge_attr=d2bond_emb3)
        atom_emb4 = self.activation(atom_emb4)
        atom_emb5,d2bond_emb5= self.gatlayer5(atom_emb4, batch_graph.edge_index, edge_attr=d2bond_emb4)
        atom_emb5 = self.activation(atom_emb5)
        total_graph2d_emb = torch.concat([atom_emb1,atom_emb2,atom_emb3,atom_emb4,atom_emb5],dim=-1)
        total_graph2d_emb = self.gat2d_ln(total_graph2d_emb)
        atom_emb5_out = self.agg1(total_graph2d_emb, index=batch_graph.batch)

        sec_stru_emb = self.sec_stru_embedding(batch_3d_graph['sec_stru_type'])
        atom_3d_emb = self.atom3d_embedding(batch_3d_graph['atom_type'])
        bond_3d_emb = self.bond3d_embedding(batch_3d_graph['edge_type'])
        atom_3d_emb = torch.concat([sec_stru_emb,atom_3d_emb],dim=-1)
        atom_3d_emb1,bond_3d_emb1 = self.gat3dlayer1(atom_3d_emb, batch_3d_graph.edge_index, edge_attr=bond_3d_emb)
        atom_3d_emb1 = self.activation(atom_3d_emb1)
        atom_3d_emb2,bond_3d_emb2 = self.gat3dlayer2(atom_3d_emb1, batch_3d_graph.edge_index, edge_attr=bond_3d_emb1)
        atom_3d_emb2 = self.activation(atom_3d_emb2)
        atom_3d_emb3,bond_3d_emb3 = self.gat3dlayer3(atom_3d_emb2, batch_3d_graph.edge_index, edge_attr=bond_3d_emb2)
        atom_3d_emb3 = self.activation(atom_3d_emb3)
        atom_3d_emb4,bond_3d_emb4 = self.gat3dlayer4(atom_3d_emb3, batch_3d_graph.edge_index, edge_attr=bond_3d_emb3)
        atom_3d_emb4 = self.activation(atom_3d_emb4)
        atom_3d_out = torch.concat([atom_3d_emb1,atom_3d_emb2,atom_3d_emb3,atom_3d_emb4],dim=-1)
        atom_3d_out = self.gat3d_ln(atom_3d_out)
        atom_3d_out = self.agg1(atom_3d_out, index=batch_3d_graph.batch)


        aac_emb_lstm = pack_padded_sequence(seq_emb,lengths=batch_seq_len,batch_first=True,enforce_sorted=False)
        aac_emb_cnn1 = pack_padded_sequence(seq_emb_2p, lengths=batch_seq_len-1, batch_first=True, enforce_sorted=False)

        _,(aac_emb_lstm1,_) = self.lstm1(aac_emb_lstm)
        aac_emb_lstm1 = aac_emb_lstm1[-4:, :, :]
        aac_emb_lstm1 = torch.permute(aac_emb_lstm1, [1, 0, 2])
        aac_emb_lstm1 = torch.flatten(aac_emb_lstm1, start_dim=1)
        aac_emb_lstm1 = self.lstm1_ln(aac_emb_lstm1)

        _,(aac_emb_cnn_lstm1,_) = self.lstm2(aac_emb_cnn1)
        aac_emb_cnn_lstm1 = aac_emb_cnn_lstm1[-4:, :, :]
        aac_emb_cnn_lstm1 = torch.permute(aac_emb_cnn_lstm1, [1, 0, 2])
        aac_emb_cnn_lstm1 = torch.flatten(aac_emb_cnn_lstm1, start_dim=1)
        aac_emb_cnn_lstm1 = self.lstm2_ln(aac_emb_cnn_lstm1)

        for trans_layer in self.transformer_layers:
            trans_emb = trans_layer(trans_emb,src_key_padding_mask=pad_mask_batch)
        trans_emb[pad_mask_batch] = 0
        trans_emb = torch.sum(trans_emb,dim=1)
        batch_seq_len = torch.unsqueeze(batch_seq_len,dim=-1).cuda()
        trans_emb = trans_emb/batch_seq_len
        batch_aac_prop = torch.concat([batch_aac_prop,batch_aac2_prop],dim=-1)
        batch_aac_prop = self.dpc_ln(batch_aac_prop)

        trans_out = self.trans_out_ln(trans_emb)
        lstm_out = self.lstm_out_ln(torch.concat([aac_emb_lstm1,aac_emb_cnn_lstm1],dim=-1))
        graph_2d_out = self.graph2d_out_ln(atom_emb5_out)
        graph_3d_out = self.graph3d_out_ln(atom_3d_out)

        total_h = torch.concat([trans_emb,aac_emb_lstm1,aac_emb_cnn_lstm1,atom_emb5_out,atom_3d_out,batch_aac_prop,batch_aac_pair],dim=-1)
        out_final = self.out_ln(total_h)
        return trans_out,lstm_out,graph_2d_out,graph_3d_out,out_final


