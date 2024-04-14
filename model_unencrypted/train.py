import torch
import torch.nn as nn
import torch.optim as optim

import time
import os
import logging
import argparse

from dataset_utils import get_data_loader
from model_class import ResNet

def train(net, epochs, batch_size, lr, reg, num_classes, device, log_every_n=500):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    
    logging.basicConfig(filename=f'resnet_train_n_classes_{num_classes}.log', level=logging.INFO, 
                        format='%(levelname)s: %(asctime)s %(message)s')
    logging.info('==> Preparing data..')
    trainloader, testloader = get_data_loader('train', batch_size, num_classes, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.875, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    global_steps = 0
    start = time.time()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        logging.info('Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                logging.info("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()

        scheduler.step()

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        logging.info("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            logging.info("Saving...")
            torch.save(net.state_dict(), f"checkpoints/resnet_{num_classes}.pt")
    return


out_chan_dict = {2: 8, 4: 10, 8: 12, 16: 14, 32: 16, 64: 18, 128: 22, 256: 24, 512: 26, 1000: 28}
if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_classes', type=int, required=True)
    args = parser.parse_args()
    num_classes = args.num_classes
    assert num_classes in out_chan_dict.keys()
    net = ResNet(num_classes, out_chan=out_chan_dict[num_classes]).to(device)
    train(net, epochs=200, batch_size=512, lr=0.01, reg=1e-4, num_classes=num_classes, device=device)
