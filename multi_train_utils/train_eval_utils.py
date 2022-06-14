import torch
from tqdm import tqdm
import sys
from multi_train_utils.distributed_utils import reduce_value, is_main_process
from multi_train_utils.test_utils import unique_image, group_image, featurize, compute_accuracy
import os.path as osp

def train_one_epoch(backbone, metric, train_loader, optimizer, epoch, device, criterion):

    backbone.train()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        train_loader = tqdm(train_loader, file=sys.stdout)

    for step, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        embeddings = backbone(data)
        thetas = metric(embeddings, labels)
        loss = criterion(thetas, labels)

        # 反向梯度信息
        loss.backward()

        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            train_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # 参数更新
        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()

@torch.no_grad()
def evaluate(model, conf, tb_writer, mean_loss, epoch, optimizer):

    model.eval()

    images = unique_image(conf.val_list)
    images = [osp.join(conf.val_root, img) for img in images]
    groups = group_image(images, conf.test_batch_size)

    feature_dict = dict()
    for group in groups:
        d = featurize(group, conf.test_transform, model, conf.device)
        feature_dict.update(d)
    accuracy, threshold = compute_accuracy(feature_dict, conf.val_list, conf.val_root)

    model_name = f"{epoch}.pth"
    print(
        f"Test Model: {model_name}\n"
        f"Accuracy: {accuracy:.3f}\n"
        f"Threshold: {threshold:.3f}\n"
    )

    tags = ["loss", "accuracy", "threshold", "learning_rate"]
    tb_writer.add_scalar(tags[0], mean_loss, epoch)
    tb_writer.add_scalar(tags[1], accuracy, epoch)
    tb_writer.add_scalar(tags[2], threshold, epoch)
    tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

    return accuracy, threshold
