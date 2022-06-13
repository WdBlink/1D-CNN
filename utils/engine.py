import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn
import utils.utils as utils
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    loss_dict = []
    loss_func = torch.nn.CrossEntropyLoss()
    for waves, targets in metric_logger.log_every(data_loader, print_freq, header):
        waves = waves.to(device)
        targets = targets.to(device)
        # waves = torch.from_numpy(waves).to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # targets = torch.tensor(np.array(targets)).to(device)
        # targets = list(target.to(device) for target in targets)

        outputs = model(waves)
        loss = loss_func(outputs, targets)
        loss_value = loss.item()
        loss_dict.append(loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(np.mean(loss_dict))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    prog_iter_test = tqdm(data_loader, desc="Testing", leave=False)
    all_pred_prob = []
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)
        all_pred_prob.append(pred.cpu().data.numpy())
    all_pred_prob = np.concatenate(all_pred_prob)
    all_pred = np.argmax(all_pred_prob, axis=1)
    return all_pred

