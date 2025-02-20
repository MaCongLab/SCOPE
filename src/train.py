import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from torch import nn
from Dataset import protein_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from SCOPE_model import model
from tqdm import tqdm
import torch_geometric as pyg
import argparse

torch.multiprocessing.set_sharing_strategy('file_system')


def loss_func(metric,trans_out,lstm_out,graph_2d_out,graph_3d_out,out_final,label):
    trans_loss = metric(trans_out,label)
    lstm_loss = metric(lstm_out, label)
    graph_2d_out_loss = metric(graph_2d_out, label)
    graph_3d_out_loss = metric(graph_3d_out,label)
    out_final = metric(out_final,label)
    return trans_loss+lstm_loss+graph_2d_out_loss+graph_3d_out_loss+out_final



def collat_fn(batch):
    batch_seq_lst = []
    batch_label_lst = []
    batch_pad_mask_lst = []
    batch_seq_len_lst = []
    batch_graph_lst = []
    batch_3d_graph_lst = []
    batch_aac_prop_lst = []
    batch_aac2_prop_lst =[]
    batch_pair_lst = []
    for data in batch:
        batch_seq, batch_pad_mask, batch_seq_len, batch_graph,batch_graph_3d, batch_aac_prop,batch_aac2_prop,batch_aac_pair,batch_label = data
        batch_seq_lst.append(batch_seq)
        batch_label_lst.append(batch_label)
        batch_pad_mask_lst.append(batch_pad_mask)
        batch_seq_len_lst.append(batch_seq_len)
        batch_graph_lst.append(batch_graph)
        batch_3d_graph_lst.append(batch_graph_3d)
        batch_aac_prop_lst.append(batch_aac_prop)
        batch_aac2_prop_lst.append(batch_aac2_prop)
        batch_pair_lst.append(batch_aac_pair)

    batch_seq = torch.vstack(batch_seq_lst)
    batch_label = torch.tensor(batch_label_lst)
    batch_pad_mask = torch.vstack(batch_pad_mask_lst)
    batch_aac_prop_lst = torch.vstack(batch_aac_prop_lst)
    batch_seq_len = torch.tensor(batch_seq_len_lst)
    batch_graph = pyg.data.Batch.from_data_list(batch_graph_lst)
    batch_graph_3d = pyg.data.Batch.from_data_list(batch_3d_graph_lst)
    batch_aac2_prop = torch.vstack(batch_aac2_prop_lst)
    batch_aac_pair = torch.vstack(batch_pair_lst)
    return batch_seq, batch_label, batch_pad_mask, batch_seq_len, batch_graph,batch_graph_3d,batch_aac_prop_lst,batch_aac2_prop,batch_aac_pair




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="name of the experiment")
    parser.add_argument("data", help="your train set path")
    parser.add_argument("save_folder", help="save model path")
    args = parser.parse_args()

    writer = SummaryWriter(f'{args.exp_name}')
    params = {'batch_size': 128,
              'shuffle': True,
              'num_workers':6}
    device = 'cuda'
    epochs = 500
    train_data_path = args.data
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    train_data_set = protein_dataset(train_data_path,maxlen=50,pos_dim=128)
    train_generator = DataLoader(train_data_set, collate_fn=collat_fn,**params)
    model = model(feat_dim=128,trans_layer_num=12).to(device)
    criteria = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(params=model.parameters(), lr=3e-4, betas=(0.9, 0.999),weight_decay=1e-3)
    global_counter = 0
    for ep in range(epochs):
        model.train()
        for batch in tqdm(train_generator):
            batch_seq,batch_label, batch_pad_mask,batch_seq_len,batch_graph,batch_graph_3d,batch_aac_prop,batch_aac2_prop,batch_aac_pair = batch
            batch_graph = batch_graph.to('cuda')
            batch_graph_3d = batch_graph_3d.to('cuda')
            label = batch_label.cuda()
            batch_pad_mask = batch_pad_mask.cuda()
            trans_out,lstm_out,graph_2d_out,graph_3d_out,out_final = model(batch_seq.cuda(), batch_pad_mask,batch_seq_len,batch_graph,batch_graph_3d,batch_aac_prop.cuda(),batch_aac2_prop.cuda(),batch_aac_pair.cuda())
            cost = loss_func(criteria,trans_out,lstm_out,graph_2d_out,graph_3d_out,out_final,label)
            tmpcost = cost.cpu().item()
            writer.add_scalar('Loss/train_celoss', tmpcost, global_counter)
            global_counter += 1
            opt.zero_grad()
            cost.backward()
            opt.step()
        torch.save(model.state_dict(), f'{save_folder}var_model_ep{ep}.ckpt')
