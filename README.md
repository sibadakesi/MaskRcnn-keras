# MaskRcnn-keras
这是由Keras编写的MaskRcnn，实现了从标定到评估，推断的功能。

# 标定方法
使用labelme工具，可以通过预先通过pip安装。

# 快速开始
1,标定使用labelme,标定结束会得到所有图片的json文件（包含标定信息）
2,将数据集切分为训练集和验证集，并且转换为训练所需要的格式,labeled_dir为使用labelme标定的结果目录，默认应该是和图片相同的目录，output_dir为转换的数据集格式的输出目录，rate为验证集的比例，训练集的比例则为1-rate。程序会在指定的目录生成train.json和val.json。同时会打印出类别对应id，从1开始，0默认为背景，最好是保存下来。
```
python labelme2COCO.py --labeled_dir XXX --output_dir XXX --rate 0.3
```
3，训练,traindata_json为转换训练集生成的train.json地址，--traindata_dir为标定的目录,也就是图片的目录
```
python train.py --traindata_json data/train.json --traindata_dir data/train --valdata_json data/val.json --valdata_dir data/train
```
4, 推断,可以将步骤1输出的类别对应id的字典复制到influence代码， --model_path模型地址，--img_path为图片地址，--show_path是推断的结果保存的地址，默认不保存。
```
python influence.py --model_path xx --img_path xx --show_path xx
```
5, 评估

# training schedule
