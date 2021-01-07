import os
import time
import numpy as np
import argparse
from config import MaskRcnnConfig


class OwnConfig(MaskRcnnConfig):
    """
    需要修改的参数
    """
    NUM_CLASSES = 23


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
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')


    args = parser.parse_args()
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    config = OwnConfig()

