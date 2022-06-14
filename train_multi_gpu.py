import os
import os.path as osp
import torch
import torch.optim as optim
from models.resnet import resnet_face18
from models.ArcFace import ArcFace
from models.loss import FocalLoss
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.datasets import Dataset
from config import Config
import tempfile
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate
from models.fmobilenet import FaceMobileNet

# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # config
    conf = Config()

    # 初始化各进程环境
    init_distributed_mode(args=conf)

    rank = conf.rank
    device = torch.device(conf.device)
    batch_size = conf.train_batch_size
    conf.lr *= conf.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    # 在第一个进程中打印信息，并实例化tensorboard
    if rank == 0:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()

    # dataset
    train_dataset, train_class_num = Dataset(conf)
    embedding_size = conf.embedding_size

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    # dataloader
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_batch_sampler,
                              pin_memory=True,
                              num_workers=nw)

    # model
    if conf.backbone == 'resnet':
        backbone = resnet_face18(use_se=conf.use_se).to(device)
    else:
        backbone = FaceMobileNet(embedding_size).to(device)

    metric = ArcFace(embedding_size, train_class_num).to(device)
    checkpoint_path_backbone = os.path.join(tempfile.gettempdir(), "initial_weights_backbone.pth")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if rank == 0:
        torch.save(backbone.state_dict(), checkpoint_path_backbone)

    dist.barrier()

    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    backbone.load_state_dict(torch.load(checkpoint_path_backbone, map_location=device))

    # 转为DDP模型
    backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[conf.gpu])
    metric = torch.nn.parallel.DistributedDataParallel(metric, device_ids=[conf.gpu])

    # optimizer criterion
    if conf.optimizer == 'sgd':
        optimizer = optim.SGD([{'params': backbone.parameters()}, {'params': metric.parameters()}],
                              lr=conf.lr, weight_decay=conf.weight_decay)
    else:
        optimizer = optim.Adam([{'params': backbone.parameters()}, {'params': metric.parameters()}],
                               lr=conf.lr, weight_decay=conf.weight_decay)

    criterion = FocalLoss(gamma=2)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

    # Checkpoints Setup
    checkpoints = conf.checkpoints
    os.makedirs(checkpoints, exist_ok=True)

    # load pretrained model
    if conf.restore:
        weights_path = osp.join(checkpoints, conf.restore_model)
        backbone.load_state_dict(torch.load(weights_path, map_location=device))

    acc = 0
    th = 0
    for epoch in range(conf.epoch):
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(backbone=backbone,
                                    metric=metric,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        if rank == 0:
            accuracy, threshold = evaluate(backbone, conf, tb_writer, mean_loss, epoch, optimizer)
            if accuracy > acc:
                backbone_path = conf.test_model
                torch.save(backbone.state_dict(), backbone_path)
                print("Saved best model.")
                acc = accuracy
                th = threshold

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()

    print("====================================================")
    print("Best Acc: {} Threshold: {}".format(acc, th))