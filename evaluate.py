import os
import json
from influence import MaskRcnn
from tqdm import tqdm

os.mkdir('groundtruths')
os.mkdir('detections')

class_names = {0: 'BG', 1: 'Ctobacco', 2: 'Btobacco', 3: 'Xtobacco'}
info = json.load(open('data/val.json', 'r'))
for inf in info['images']:
    file_name = inf['file_name']
    id = inf['id']
    with open(os.path.join('groundtruths', os.path.splitext(file_name)[0] + '.txt'), 'w') as f:
        for annotation in info['annotations']:
            if id == annotation['image_id']:
                f.write(class_names[annotation['category_id']])
                for crood in annotation['bbox']:
                    f.write(' ')
                    f.write(str(round(crood)))
                f.write('\n')

maskrcnn = MaskRcnn()
maskrcnn.init_app(r'logs\20211123T1226\mask_rcnn_0011.h5', class_names)  # path

info = json.load(open('data/val.json', 'r'))['images']
for img_dict in tqdm(info):
    path = os.path.join('before', img_dict['file_name'])
    result = maskrcnn.predict(path, flag=False)
    with open(os.path.join('detections', os.path.splitext(img_dict['file_name'])[0] + '.txt'), 'w') as f:
        for i in range(len(result['class_ids'])):
            f.write(class_names[result['class_ids'][i]])
            f.write(' ')
            f.write(str(result['scores'][i]))
            f.write(' ')
            f.write(str(result['rois'][i][1]))
            f.write(' ')
            f.write(str(result['rois'][i][0]))
            f.write(' ')
            f.write(str(result['rois'][i][3]))
            f.write(' ')
            f.write(str(result['rois'][i][2]))
            f.write('\n')
