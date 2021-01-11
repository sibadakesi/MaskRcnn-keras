import os
import time
import numpy as np
import argparse
from config import MaskRcnnConfig
from dataset import OwnDataset
import modelibe

class OwnConfig(MaskRcnnConfig):
    """
    需要修改的参数
    """
    NUM_CLASSES = 23  # 根据自己的训练集类别。包含背景，所以为实际类别+1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN.')

    parser.add_argument('--logs', required=False,
                        default='logs',
                        help='Logs and checkpoints directory (default=logs/)')

    parser.add_argument('--traindata_json', required=True,
                        help='path to train dataset json file')
    parser.add_argument('--traindata_dir', required=True, help='path to train dataset dir ')

    parser.add_argument('--valdata_json', required=True,
                        help='path to  Validation dataset json file')
    parser.add_argument('--valdata_dir', required=True, help='path to Validation dataset dir ')

    parser.add_argument('--pretrain_path', required=False, default=None)

    args = parser.parse_args()

    print("Logs: ", args.logs)

    config = OwnConfig()

    model = modelibe.MaskRcnn(mode="training", config=config,
                              model_dir=args.logs)
    if args.pretrain_path is not None:
        model.load_weights('mask_rcnn_coco.h5', by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])  # 这个是由coco数据集训练得出的，如果用自己的训练集，只能载入部分

    dataset_train = OwnDataset()
    dataset_train.load_own(args.traindata_json, args.traindata_dir)
    dataset_train.prepare()

    dataset_val = OwnDataset()
    dataset_val.load_own(args.valdata_json, args.valdata_dir)
    dataset_val.prepare()
    from modelibe import compose_image_meta
    # training schedule ,分别是训练头部，提取特征部分，以及全部训练，全部训练将学习率缩小十倍，具体可以自行修改
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=24,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=34,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=44,
                layers='all')




