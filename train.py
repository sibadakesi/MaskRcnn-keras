import os
import time
import numpy as np
import argparse
from config import MaskRcnnConfig
import modelibe

class OwnConfig(MaskRcnnConfig):
    """
    需要修改的参数
    """
    NUM_CLASSES = 23  # 根据自己的训练集类别。包含背景，所以为实际类别+1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN.')
    parser.add_argument('--dataset', required=False,
                        metavar="",
                        help='数据地址')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default='logs',
                        help='Logs and checkpoints directory (default=logs/)')
    prrser.add_argument('--pretrain_path',required=False,default=True)


    args = parser.parse_args()
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    config = OwnConfig()

    model = modelibe.MaskRcnn(mode="training", config=config,
                              model_dir=args.logs)

    model.load_weights('mask_rcnn_coco.h5', by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])  # 这个是由coco数据集训练得出的，如果用自己的训练集，只能载入部分

    dataset_train = OwnDataset()