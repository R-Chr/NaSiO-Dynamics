import torch
import numpy as np

def train(model, optimizer, loss_fn, train_loader, device):
    model.train()
    loss_list = []
    pearson_list = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    return sum(loss_list) / len(loss_list)

def validate(model, loss_fn, val_loader, device):
    model.eval()
    with torch.no_grad():
        loss_list = []
        pearson_list = []
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y.unsqueeze(1))
            loss_list.append(loss.item())

    return sum(loss_list) / len(loss_list)

def load_ckp(checkpoint_fpath, model, optimizer, scheduler, device):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch']

def save_ckp(state, is_best):
    torch.save(state, 'checkpoint.pt')
    if is_best:
        torch.save(state,'best_model.pt')

def PearsonCC(output, target):
    x = output
    y = target
    vx = x - np.mean(x, axis=0)
    vy = y - np.mean(y, axis=0)
    cost = np.sum(vx * vy, axis=0) / (np.sqrt(np.sum(vx ** 2, axis=0)) * np.sqrt(np.sum(vy ** 2, axis=0)))
    return np.mean(cost)

