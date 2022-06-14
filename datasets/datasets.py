from torchvision.datasets import ImageFolder

def Dataset(conf):
    """
    :param conf:
    :return: train dataset and class_num
    """

    dataroot = conf.train_root
    transform = conf.train_transform

    data = ImageFolder(dataroot, transform=transform)
    class_num = len(data.classes)

    return data, class_num