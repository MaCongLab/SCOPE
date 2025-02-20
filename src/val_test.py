import pandas as pd
import torch
from Dataset import protein_dataset
from torch.utils.data import DataLoader
from SCOPE_model import model
import torch_geometric as pyg
from utils import SP,SN,FDR,ACC,MCC

torch.multiprocessing.set_sharing_strategy('file_system')


def collat_fn(batch):
    batch_seq_lst = []
    batch_pad_mask_lst = []
    batch_seq_len_lst = []
    batch_graph_lst = []
    batch_3d_graph_lst = []
    batch_aac_prop_lst = []
    batch_aac2_prop_lst =[]
    batch_pair_lst = []
    batch_label_lst = []
    # batch_bond_graph_lst = []
    for data in batch:
        batch_seq, batch_pad_mask, batch_seq_len, batch_graph,batch_graph_3d, batch_aac_prop,batch_aac2_prop,batch_aac_pair,batch_label = data
        batch_seq_lst.append(batch_seq)
        batch_pad_mask_lst.append(batch_pad_mask)
        batch_seq_len_lst.append(batch_seq_len)
        batch_graph_lst.append(batch_graph)
        batch_3d_graph_lst.append(batch_graph_3d)
        batch_aac_prop_lst.append(batch_aac_prop)
        batch_aac2_prop_lst.append(batch_aac2_prop)
        batch_pair_lst.append(batch_aac_pair)
        batch_label_lst.append(batch_label)
        # batch_bond_graph_lst.append(batch_bond_graph)
    # print(batch_seq_lst)
    batch_seq = torch.vstack(batch_seq_lst)
    batch_pad_mask = torch.vstack(batch_pad_mask_lst)
    batch_aac_prop_lst = torch.vstack(batch_aac_prop_lst)
    batch_seq_len = torch.tensor(batch_seq_len_lst)
    batch_graph = pyg.data.Batch.from_data_list(batch_graph_lst)
    batch_graph_3d = pyg.data.Batch.from_data_list(batch_3d_graph_lst)
    batch_aac2_prop = torch.vstack(batch_aac2_prop_lst)
    batch_aac_pair = torch.vstack(batch_pair_lst)
    batch_label = torch.tensor(batch_label_lst)
    return batch_seq, batch_pad_mask, batch_seq_len, batch_graph,batch_graph_3d,batch_aac_prop_lst,batch_aac2_prop,batch_aac_pair,batch_label




if __name__ == '__main__':
    test_params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 1}
    device = 'cuda'
    test_data_path = "../data/test.csv"

    test_data_set = protein_dataset(test_data_path,maxlen=50,pos_dim=128)
    test_generator = DataLoader(test_data_set, collate_fn=collat_fn, **test_params)
    model = model(feat_dim=128,trans_layer_num=12).to(device)
    df = pd.read_csv(test_data_path)
    model.load_state_dict(
        torch.load('../save_model/scope_model.ckpt', map_location=device),
        strict=True)

    model.eval()
    pred_lst = []
    label_lst = []
    with torch.no_grad():
        total_num = 0
        hit_len = 0
        for batch in test_generator:
            batch_seq, batch_pad_mask, batch_seq_len, batch_graph, batch_graph_3d,batch_aac_prop, batch_aac2_prop,batch_aac_pair,batch_label = batch
            batch_graph = batch_graph.to('cuda')
            batch_graph_3d = batch_graph_3d.to('cuda')
            batch_pad_mask = batch_pad_mask.cuda()
            batch_label = batch_label.cuda()
            trans_out,lstm_out,graph_2d_out,graph_3d_out,out_final = model(batch_seq.cuda(),batch_pad_mask,batch_seq_len,batch_graph,batch_graph_3d,batch_aac_prop.cuda(),batch_aac2_prop.cuda(),batch_aac_pair.cuda())
            trans_out = torch.softmax(trans_out,dim=-1)
            lstm_out = torch.softmax(lstm_out,dim=-1)
            graph_2d_out = torch.softmax(graph_2d_out, dim=-1)
            graph_3d_out = torch.softmax(graph_3d_out, dim=-1)
            out_final = torch.softmax(out_final,dim=-1)
            total_out = out_final+graph_2d_out+graph_3d_out+trans_out+lstm_out
            pred = torch.argmax(total_out,dim=-1)
            pred = pred.cpu().numpy().tolist()
            batch_label = batch_label.cpu().numpy().tolist()
            label_lst += batch_label
            pred_lst = pred_lst+pred


        print('SN: ',SN(pred_lst,label_lst))
        print('SP: ', SP(pred_lst, label_lst))
        print('FDR: ',FDR(pred_lst,label_lst))
        print('ACC: ',ACC(pred_lst,label_lst))
        print('MCC: ', MCC(pred_lst,label_lst))