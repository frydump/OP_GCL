import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
from loader import MoleculeDataset_graphcl
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import GNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)


class graphcl(nn.Module):
    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def Get_Max_Min(self,s_m_1,s_m_2):
        ab = torch.abs(s_m_1-s_m_2)
        sum = s_m_1+s_m_2
        s_m_max = (sum + ab)/2
        s_m_min = (sum - ab)/2
        return s_m_max,s_m_min

    def loss_cl(self, x, x1, x2, Loss=1,mean=True, beta=0.3):
        T = 0.5
        batch_size, _ = x1.size()

        x_abs = x.norm(dim=1)
        x1_abs = x1.norm(dim=1)  # 求范数
        x2_abs = x2.norm(dim=1)

        # 爱因斯坦求和简记法，求A*B/（范数的乘积），也就是余弦相似度，不过这个未必有内嵌函数的运行效率高吧
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_1 = torch.einsum('ik,jk->ij', x, x1) / torch.einsum('i,j->ij', x_abs, x1_abs)
        sim_matrix_2 = torch.einsum('ik,jk->ij', x, x2) / torch.einsum('i,j->ij', x_abs, x2_abs)
        sim_matrix_0 = torch.einsum('ik,jk->ij', x, x) / torch.einsum('i,j->ij', x_abs, x_abs)

        s_m_1 = torch.diag(sim_matrix_1).reshape(-1, 1).to(device)
        s_m_2 = torch.diag(sim_matrix_2).reshape(-1, 1).to(device)
        s_m_max, s_m_0 = self.Get_Max_Min(s_m_1, s_m_2)

        MinSLB = s_m_0.mean().detach()  # 实验标签
        Sub = (s_m_max - s_m_0).mean().detach()
        DSum = (1 - sim_matrix_0).sum(1).mean().detach()

        sim_matrix = torch.exp(sim_matrix / T)
        sim_matrix_1 = torch.exp(sim_matrix_1 / T)  # 不行的话就把分子加上吧
        sim_matrix_2 = torch.exp(sim_matrix_2 / T)

        pos_sim = torch.diag(sim_matrix).reshape(-1, 1).to(device)
        s_m_1 = torch.diag(sim_matrix_1).reshape(-1, 1).to(device)
        s_m_2 = torch.diag(sim_matrix_2).reshape(-1, 1).to(device)
        s_m_max, s_m_0 = self.Get_Max_Min(s_m_1, s_m_2)


        sim_matrix_1 = sim_matrix_1.masked_fill(sim_matrix_1 == s_m_max.expand(sim_matrix_1.size()), 1e-7)#不加掩码
        sim_matrix_2 = sim_matrix_2.masked_fill(sim_matrix_2 == s_m_max.expand(sim_matrix_2.size()), 1e-7)
        # sim_matrix_1 = sim_matrix_1.masked_fill(sim_matrix_1 > s_m_0.expand(sim_matrix_1.size()), 1e-7)  # 增加掩码
        # sim_matrix_2 = sim_matrix_2.masked_fill(sim_matrix_2 > s_m_0.expand(sim_matrix_2.size()), 1e-7)
        # sim_matrix = sim_matrix.masked_fill(sim_matrix > pos_sim.expand(sim_matrix.size()), 1e-7)#当采用掩码时不再需要-pos

        if Loss%3 == 1:
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  # 上半/下半

        elif Loss%3 == 2:
            pos_sim = pos_sim.masked_fill(pos_sim < s_m_0, 7.3891)  # 如果在同一个方向，直接拉进最远与原图
            loss = s_m_0.T / (sim_matrix_1.sum(dim=1) + sim_matrix_2.sum(dim=1) - s_m_0)#是不是因为分母太小导致出现NAN
            loss = - torch.log(loss)
            if mean:
                loss = loss.mean()
            return loss + 1 / pos_sim.mean(), [MinSLB, Sub, DSum]
        else:
            s_m_max = s_m_max.masked_fill(pos_sim <= s_m_0.mean(), 1e-7)  # 全程拉近min，当相似时拉近max，不加尾项
            loss = torch.pow(s_m_0.T, 1) / (
                    torch.pow((sim_matrix_1.sum(dim=1) + sim_matrix_2.sum(dim=1) - s_m_0), 1 - beta) * torch.pow(
                s_m_max.T, beta))  # 快速收敛，拉小关键和边缘的差距

        loss = - torch.log(loss)

        if mean:
            loss = loss.mean()

        return loss, [MinSLB, Sub, DSum]
        # T = 0.1
        # batch_size, _ = x1.size()
        # x1_abs = x1.norm(dim=1)
        # x2_abs = x2.norm(dim=1)
        #
        # sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        # sim_matrix = torch.exp(sim_matrix / T)
        # pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        # loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        # loss = - torch.log(loss).mean()
        # return loss

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def train(args, loader, model, optimizer, device, loss_num):
    model.train()
    total_loss = 0
    label_lists = []
    for data, data1, data2 in loader:
        # if len(data.y) < 128:  # 将极端少的批次踢掉
        #     break
        # print(data1, data2)
        optimizer.zero_grad()
        data = data.to(device)
        data1 = data1.to(device)
        data2 = data2.to(device)
        out = model.forward_cl(data.x, data.edge_index, data.edge_attr, data.batch)
        out1 = model.forward_cl(data1.x, data1.edge_index, data1.edge_attr, data1.batch)
        out2 = model.forward_cl(data2.x, data2.edge_index, data2.edge_attr, data2.batch)

        loss, label = model.loss_cl(out, out1, out2,loss_num)
        loss.backward()
        label_lists.append(label)
        total_loss += float(loss.detach().cpu().item())
        optimizer.step()

    return total_loss / len(loader.dataset), torch.tensor(label_lists).mean(0)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')

    parser.add_argument('--aug_mode', type=str, default = 'sample') 
    parser.add_argument('--aug_strength', type=float, default = 0.2)
    parser.add_argument('--loss_num', type=float, default=1)
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    dataset = MoleculeDataset_graphcl("dataset/" + args.dataset, dataset=args.dataset)
    dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)


    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = graphcl(gnn)
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    aug_prob = np.ones(25) / 25
    label = 0
    for epoch in range(1, args.epochs+1):
        dataset.set_augProb(aug_prob)
        pretrain_loss, label_list= train(args, loader, model, optimizer, device, args.loss_num)
        label = label + label_list
        print(epoch, pretrain_loss)
        if epoch % 5 == 0:
            torch.save(model.gnn.state_dict(), "./weights/" +str(int(args.loss_num))+"_"+ str(args.loss_num) + '_' + str(epoch) + ".pth")
            print(label / 5)
            label = 0


if __name__ == "__main__":
    main()

