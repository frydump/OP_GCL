import torch
from torch.optim import Adam
from tu_dataset import DataLoader
from utils import print_weights

torch.multiprocessing.set_sharing_strategy('file_system')#防止dataloader加载数据时出现的错误

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def experiment(dataset, model_func, epochs, batch_size, lr, weight_decay,
                dataset_name=None, aug_mode='uniform', aug_ratio=0.2, beta = 0.3, suffix=0):
    model = model_func(dataset).to(device)
    print_weights(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dataset.set_aug_mode(aug_mode)
    dataset.set_aug_ratio(aug_ratio)

    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=16)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    label = 0
    for epoch in range(1, epochs+1):
        pretrain_loss,label_list = train(loader, model, optimizer, device, beta)
        label = label+label_list
        print(pretrain_loss)
        if epoch % 20 == 0:
            weight_path = './weights_graphcl/' + dataset_name + '_' + str(lr) + '_' + str(epoch) + '_' + str(suffix)  + '.pt'
            torch.save(model.state_dict(), weight_path)
            print(label/20)
            label=0


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(loader, model, optimizer, device, beta):
    model.train()
    total_loss = 0
    label_lists =[]
    for data, data1, data2 in loader:
        # if len(data.y)< 128:#将极端少的批次踢掉
        #     break
        optimizer.zero_grad()
        data = data.to(device)
        data1 = data1.to(device)
        data2 = data2.to(device)
        out = model.forward_graphcl(data)
        out1 = model.forward_graphcl(data1)
        out2 = model.forward_graphcl(data2)

        loss,label= model.loss_graphcl(out, out1, out2, beta = beta)
        loss.backward()
        label_lists.append(label)
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()

    return total_loss/len(loader.dataset),torch.tensor(label_lists).mean(0)

