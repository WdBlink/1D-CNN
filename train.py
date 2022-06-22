import torch
import os
import cv2
import utils.transforms as T
from utils.engine import train_one_epoch, evaluate
import utils.utils as utils
import torchvision
import argparse
import numpy as np
from collections import Counter
import pickle
from dataset.dataloaders import AFDataset, MyDataset
from model.resnet1d import ResNet1D
from sklearn.metrics import classification_report
from utils.utils import read_data_hsi_4_with_val, read_data_physionet_4_with_val
from osgeo import gdal


def get_object_detection_model(num_classes):
    return num_classes


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # 50%的概率水平翻转
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


def showbbox(model, img, result_path, device):
    # 输入的img是0-1范围的tensor
    model.eval()
    with torch.no_grad():
        '''
        prediction形如：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])

    print(prediction)

    img = img.permute(1, 2, 0)  # C,H,W → H,W,C，用来画图
    img = (img * 255).byte().data.cpu()  # * 255，float转0-255
    img = np.array(img).copy()  # tensor → ndarray

    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())

        label = prediction[0]['labels'][i].item()

        if label == 1:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
            cv2.putText(img, 'mark_type_1', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                        thickness=2)
        elif label == 2:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
            cv2.putText(img, 'mark_type_2', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        thickness=2)
    cv2.imwrite(result_path, img)


def main(opt):
    root = os.path.curdir

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    # 3 classes, mark_type_1，mark_type_2，background
    num_classes = 10

    X_train, X_val, X_test, Y_train, Y_val, Y_test = read_data_hsi_4_with_val()
    train_dataset = MyDataset(X_train, Y_train)
    val_dataset = MyDataset(X_val, Y_val)
    test_dataset = MyDataset(X_test, Y_test)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # get the model using our helper function
    model = ResNet1D(in_channels=1,
                     base_filters=64,
                     kernel_size=16,
                     stride=2,
                     groups=16,
                     n_block=2,
                     n_classes=num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # SGD
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # and a learning rate scheduler
    # cos学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # let's train it for   epochs
    num_epochs = 50
    # save_path = os.path.join(root, opt.project, opt.name + '.pkl')
    save_dir = os.path.join(os.getcwd(), opt.project)
    if os.path.exists(save_dir):
        save_path = os.path.join(save_dir+'AF.pkl')
    else:
        os.makedirs(save_dir)
        save_path = os.path.join(save_dir + 'AF.pkl')
    # save_path = '/mnt/e/github/object-detection-zoo/runs/train/tmp/faster_rcnn.pkl'
    result_path = '/mnt/e/github/object-detection-zoo/runs/train/tmp/faster_rcnn.jpg'

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # engine.py的train_one_epoch函数将images和targets都.to(device)了
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the val dataset
        all_pred = evaluate(model, data_loader_val, device=device)
        print(classification_report(all_pred, Y_val))

        # save model file
        print(f'save model to {save_path}')
        torch.save(model, save_path)
        print('==================================================')
        print('')

    print("That's it!")
    print('==================================================')
    print('====================begin test====================')
    model = torch.load(save_path)
    print(device)
    model.to(device)
    test_pred = evaluate(model, data_loader_test, device=device)
    print(classification_report(test_pred, Y_test))
    # dataset_test = VOCDataset(root, get_transform(train=False))
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])
    # img, _ = dataset_test[0]
    # showbbox(model, img, result_path=result_path, devive=device)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='runs/classification/Resnet1d/last.pkl',
                        help='model.pt path(s)')
    parser.add_argument('--root', type=str, default='/mnt/e/Dataset/taining_data_2021-08-19',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1000, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-img', default=True, action='store_true', help='save img results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs\\classification\\Resnet1d', help='save results to project/name')
    parser.add_argument('--name', default='whu-hi', help='save results to project/name')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)