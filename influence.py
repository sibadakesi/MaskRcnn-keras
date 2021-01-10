# 用于推断
from config import MaskRcnnConfig
import modelibe

import tensorflow as tf
import skimage.io as io
import scipy.misc
import os
import numpy as np
import keras.backend.tensorflow_backend as KTF
from tqdm import tqdm
import cv2
import colorsys
from skimage.measure import find_contours
import argparse


class OurConfig(MaskRcnnConfig):
    NUM_CLASSES = 23  # 根据自己的训练集类别。包含背景，所以为实际类别+1
    DETECTION_MIN_CONFIDENCE = 0.5
    RPN_NMS_THRESHOLD = 0.5


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


class MaskRcnn(object):

    def init_app(self, model_path, class_names):
        self.g = tf.Graph()
        with self.g.as_default():
            config = OurConfig()
            self.model = modelibe.MaskRcnn(mode="inference", model_dir="log", config=config)
            self.model.load_weights(model_path, by_name=True)
        self.class_names = class_names

    def predict(self, path, save_path=None):
        with self.g.as_default():
            images_batch = []
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_batch.append(img)
            results = self.model.detect(images_batch, verbose=0)[0]

            self.save_show_v2(save_path, img, results['rois'], results['masks'], results['class_ids'],
                              results['scores'])
            return results

    def save_show_v2(self, show_path, image, boxes, masks, class_ids,
                     scores=None, allow=None):

        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        colors = random_colors(N)

        for i in range(N):
            class_id = class_ids[i]
            if allow:
                if class_id not in allow:
                    continue

            color = np.array(list(colors[i]))[..., ::-1].tolist()

            new_color = tuple(color)

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Label

            score = scores[i] if scores is not None else None
            # label = class_names[class_id]
            label = self.class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label

            cv2.putText(image, caption, (x1, y1 + 8), font, 0.5, (255, 255, 255), 1)

            # Mask
            mask = masks[:, :, i]
            image = apply_mask(image, mask, new_color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                verts = np.array([verts.astype(np.int32)])
                image = cv2.polylines(image, verts, True, (0, 255, 0))

        cv2.imshow("influence", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if show_path:
            masked_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(show_path, masked_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mask R-CNN influence')

    parser.add_argument('--model_path', required=True,
                        help='.h5 file ')

    parser.add_argument('--img_path', required=True,
                        help='path to img')

    parser.add_argument('--show_path', required=False, default=None)

    args = parser.parse_args()

    print("model: ", args.model_path)
    print("img_path", args.img_path)

    class_names = {0: 'BG', 1: 'light', 2: 'treepolo', 3: 'markpole', 4: 'car', 5: 'electricalbox',
                   6: 'voltagetransformer', 7: 'bicycle', 8: 'person', 9: 'telegraphpole', 10: 'garbage',
                   11: 'airconditioner', 12: 'motorcycle', 13: 'shopsign', 14: 'othermark', 15: 'Independentsign',
                   16: ' truck', 17: 'trafficlight', 18: 'telephonebox', 19: 'stopmark', 20: 'bus', 21: 'bench', 22
                   : 'subwaymark'}

    maskrcnn = MaskRcnn()
    maskrcnn.init_app(class_names, args.model_path)
    maskrcnn.predict(args.img_path, save_path=args.show_path)
