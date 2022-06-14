import torch
import torchvision.transforms as T


class Config(object):

    backbone = 'resnet'

    # network settings
    embedding_size = 512
    drop_ratio = 0.5
    use_se = False

    # data preprocess
    input_shape = [1, 128, 128]
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomHorizontalFlip(),
        T.Resize((144, 144)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    # dataset
    all_data = './data/105_classes_pins_dataset'

    train_root = './data/train'
    val_root = "./data/val"
    val_list = "./data/pairs.txt"

    test_root = './data/test2'
    label_root = './data/answer2.csv'

    predict_root = './data/predict'

    # training settings
    checkpoints = "checkpoints"
    restore = True
    restore_model = "resnet18_110.pth"
    test_model = "checkpoints/best_model_resnet.pth"

    train_batch_size = 64
    test_batch_size = 60

    epoch = 100
    optimizer = 'sgd'  # ['sgd', 'adam']
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss'  # ['focal_loss', 'cross_entropy']

    # multi gpu info
    device = 'cuda'
    rank = 0
    world_size = 1
    gpu = 5
    distributed = False
    dist_backend = 'nccl'
    dist_url = 'env://'

    # predict output info
    student_id = '21451943012'