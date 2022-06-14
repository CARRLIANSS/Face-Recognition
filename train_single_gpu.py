import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.resnet import resnet_face18
from models.fmobilenet import FaceMobileNet
from models.ArcFace import ArcFace
from models.loss import FocalLoss
from datasets.datasets import Dataset
from config import Config
from torch.utils.tensorboard import SummaryWriter
from multi_train_utils.train_eval_utils import evaluate
from torch.utils.data import DataLoader


if __name__ == '__main__':
    conf = Config()

    # tensorboard
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    # Data Setup
    data, class_num = Dataset(conf)
    dataloader = DataLoader(data, batch_size=conf.train_batch_size, shuffle=True,
            pin_memory=True, num_workers=8)
    embedding_size = conf.embedding_size
    device = conf.device

    # Network Setup
    if conf.backbone == 'resnet':
        net = resnet_face18(use_se=conf.use_se).to(device)
    else:
        net = FaceMobileNet(embedding_size).to(device)

    metric = ArcFace(embedding_size, class_num).to(device)

    net = nn.DataParallel(net)
    metric = nn.DataParallel(metric)

    # Training Setup
    criterion = FocalLoss(gamma=2)

    if conf.optimizer == 'sgd':
        optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}],
                              lr=conf.lr, weight_decay=conf.weight_decay)
    else:
        optimizer = optim.Adam([{'params': net.parameters()}, {'params': metric.parameters()}],
                               lr=conf.lr, weight_decay=conf.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

    # Checkpoints Setup
    checkpoints = conf.checkpoints
    os.makedirs(checkpoints, exist_ok=True)

    # load pretrained model
    if conf.restore:
        weights_path = osp.join(checkpoints, conf.restore_model)
        net.load_state_dict(torch.load(weights_path, map_location=device))

    # Start training
    net.train()

    acc = 0
    th = 0
    for e in range(conf.epoch):
        for data, labels in tqdm(dataloader, desc=f"Epoch {e}/{conf.epoch}",
                                 ascii=True, total=len(dataloader)):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = net(data)
            thetas = metric(embeddings, labels)
            loss = criterion(thetas, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {e}/{conf.epoch}, Loss: {loss}")
        accuracy, threshold = evaluate(net, conf, tb_writer, loss, e, optimizer)

        # save best model and it's threshold
        if accuracy > acc:
            backbone_path = conf.test_model
            torch.save(net.state_dict(), backbone_path)
            print("Saved best model.")
            acc = accuracy
            th = threshold

        scheduler.step()
        net.train()

    print("====================================================")
    print("Best Acc: {} Threshold: {}".format(acc, th))