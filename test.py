import dill
import numpy as np
import contflame.data.datasets as datasets
import torch
from torch import nn
from contflame.data.utils import MultiLoader
from torch.utils.data import DataLoader

import models

def train(model, optimizer, criterion, train_loader, config):
    model.train()

    correct = 0
    loss_sum = 0
    tot = 0

    for step, (data, targets) in enumerate(train_loader):
        data = data.to(config['device'])
        targets = targets.to(config['device'])
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_sum += loss.item() * data.size(0)
        tot += data.size(0)
        correct += preds.eq(targets).sum().item()

    accuracy = correct / tot
    loss = loss_sum / tot

    return loss, accuracy

def test(model, criterion, test_loader, config):
    model.eval()

    correct = 0
    loss_sum = 0
    tot = 0

    for step, (data, targets) in enumerate(test_loader):
        data = data.to(config['device'])
        targets = targets.to(config['device'])

        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        loss_sum += loss.item() * data.size(0)
        tot += data.size(0)
        correct += preds.eq(targets).sum().item()

    accuracy = correct / len(test_loader.dataset)
    loss = loss_sum / tot

    return loss, accuracy

if __name__ == '__main__':
    with open('distilled2.ptc', 'rb') as file:
        checkpoint = dill.load(file)

    config = checkpoint['config']

    run_config = config['run_config']
    model_config = config['model_config']
    param_config = config['param_config']
    data_config = config['data_config']
    log_config = config['log_config']

    criterion = nn.CrossEntropyLoss()

    net = getattr(model, model_config['arch']).Model(model_config)
    net.load_state_dict(checkpoint['init'])
    net.to(run_config['device'])

    Dataset = getattr(datasets, data_config['dataset'])
    testset = Dataset(dset='test', transform=data_config['test_transform'])
    testloader = DataLoader(testset, batch_size=256, shuffle=False, pin_memory=True, num_workers=data_config['num_workers'])


    buffer = checkpoint['dataset']
    lrs = checkpoint['lrs']

    bufferloader = MultiLoader([buffer], batch_size=len(buffer))

    for epoch in range(param_config['epochs']):
        optimizer = torch.optim.SGD(net.parameters(), lr=lrs[epoch] if epoch < len(lrs) else lrs[-1], )

        buffer_loss, buffer_accuracy = train(net, optimizer, criterion, bufferloader, run_config)
        test_loss, test_accuracy = test(net, criterion, testloader, run_config)

        metrics = {f'Test loss': test_loss,
                   f'Test accuracy': test_accuracy,
                   f'Buffer loss': buffer_loss,
                   f'Buffer accuracy': buffer_accuracy,
                   f'Epoch': epoch}
        print(metrics)