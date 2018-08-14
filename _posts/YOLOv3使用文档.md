## YOLOv3使用文档  
### 一、YOLOv3的配置  
参考：https://pjreddie.com/darknet/yolo/  
```
git clone https://github.com/pjreddie/darknet
cd darknet
```
修改Makefile文件：
```
GPU=1  #使用GPU
CUDNN=1  #使用CUDNN加速  
OPENCV=1 #使用opencv
```
```
make -j8
```
可测试一下
```
wget https://pjreddie.com/media/files/yolov3.weights
```
```
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

### 二、数据集的准备  
参考：https://blog.csdn.net/lilai619/article/details/79695109  
https://blog.csdn.net/john_bh/article/details/80625220  
#### 1.将自己的数据集转化为voc格式  
这个数据集的可放在任意位置，我是在根目录下建了一个data文件夹用来放置数据。
```
data
—— VOCdevkit
———— headdata
—————— Annotations
—————— ImageSets
———————— Main
—————— JPEGImages
```
名字可以更改，但后面的路径要写对。  
Annotations里面是.xml文件；  
Main中是4个txt文件，其中test.txt是测试集，train.txt是训练集，val.txt是验证集，trainval.txt是训练和验证集;  
JPEGImages中是所有的训练图片。
#### 2.生成YOLOv3使用的数据格式  
YOLOv3使用的是.txt数据格式，.txt里面如下图：
![](/home/rd301/文档/学习/深度学习/目标检测/YOLOv3_txt.png)  
将darknet/scripts/voc_label.py复制到data目录下，跟VOCdevkit同目录。
修改里面的内容，参考我的[voc_label.py]()：
```python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('train'), ('val'), ('test')]              #当你的Main下文件不需要year时，请去掉全文中所有year   

classes = ["person"]                              #修改为自己的类别


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('VOCdevkit/headdata/Annotations/%s.xml'%(image_id))                            # 修改为自己.xml文件路径
    out_file = open('VOCdevkit/headdata/labels/%s.txt'%(image_id), 'w')                           # 修改label存放路径
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for image_set in sets:
    if not os.path.exists('VOCdevkit/headdata/labels/'):                                                   # 修改路径
        os.makedirs('VOCdevkit/headdata/labels/')
    image_ids = open('VOCdevkit/headdata/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()        # 修改路径
    list_file = open('%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        print(image_id)
        list_file.write('%s/VOCdevkit/headdata/JPEGImages/%s.jpg\n'%(wd, image_id))                        #修改路径
        convert_annotation(image_id)
    list_file.close()

os.system("cat train.txt val.txt > train1.txt")                                                            # 将文件合并为一个文件，最好文件名都不相同，生成后再把train1.txt改回train.txt
os.system("cat train.txt val.txt test.txt > train.all.txt")                                                # 

```
生成YOLOv3训练所用数据
```
python voc_label.py 
```
### 三、训练
#### 0.修改detector.c  
目的1：便于以后计算recall  
目的2：训练之前计算自己数据的anchor大小，从而更改cfg文件
```
./darknet detector calc_anchors cfg/voc.data -num_of_clusters 9 -width 416 -height 416 -show 1  
```
来聚类anchor,然后修改cfg/yolov3-train.cfg中的anchor，修改格式可参考我的[detector.c]()
#### 1.下载预训练模型  
```
wget https://pjreddie.com/media/files/darknet53.conv.74
```
#### 2.修改配置文件
（1）修改cfg/voc.data  
```
classes= 1                                              # 类别数，我的只有一类
train  = /home/data/datahead/train.txt       #生成的YOLOv3使用的train.txt
valid  = /home/data/datahead/test.txt        #生成的test.txt
names = /home/data/datahead/headdata.names   #自己建的.name，里面是具体类别
backup = /home/darknet/backup                #存储权重的文件夹

```
（2）修改.name文件  
一行放一个类别名  
（3）修改cfg/yolov3-voc.cfg
<font color=red>这里可以将yolov3-voc.cfg备份成两个文件，yolov3-train.cfg与yolov3-test.cfg,然后训练文件中只开启train,测试文件中只开启test，且将batch=1，subdivisions=1</font>
```
[net]
# Testing            ### 测试模式                                          
# batch=1
# subdivisions=1
# Training           ### 训练模式，每次前向的图片数目 = batch/subdivisions 
batch=64
subdivisions=16
width=416            ### 网络的输入宽、高、通道数
height=416
channels=3
momentum=0.9         ### 动量 
decay=0.0005         ### 权重衰减
angle=0
saturation = 1.5     ### 饱和度
exposure = 1.5       ### 曝光度 
hue=.1               ### 色调
learning_rate=0.001  ### 学习率 
burn_in=1000         ### 学习率控制的参数
max_batches = 50200  ### 迭代次数                                          
policy=steps         ### 学习率策略 
steps=40000,45000    ### 学习率变动步长 
scales=.1,.1         ### 学习率变动因子  



[convolutional]
batch_normalize=1    ### BN
filters=32           ### 卷积核数目
size=3               ### 卷积核尺寸
stride=1             ### 卷积核步长
pad=1                ### pad
activation=leaky     ### 激活函数

......

[convolutional]
size=1
stride=1
pad=1
filters=45  #3*(10+4+1)
activation=linear

[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=10  #类别   
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=0  #1，如果显存很小，将random设置为0，关闭多尺度训练；
......

[convolutional]
size=1
stride=1
pad=1
filters=45  #3*(10+4+1)
activation=linear

[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=10  #类别
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=0  #1，如果显存很小，将random设置为0，关闭多尺度训练；
......

[convolutional]
size=1
stride=1
pad=1
filters=45  #3*(10+4+1)
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=10  #类别
num=9
jitter=.3  # 数据扩充的抖动操作
ignore_thresh = .5  #文章中的阈值1
truth_thresh = 1  #文章中的阈值2
random=0  #1，如果显存很小，将random设置为0，关闭多尺度训练；
```
必须要改的地方，classes的数目为你的类别数。然后根据你classes类别的数目修改最后一层卷积数，即[yolo]前面的一层卷积中filters。计算公式如下：
$$
N*N*[3*(4+1+C)]
$$
N=1，具体代表什么不确定。  
3，每组mask的数目，mask一共9个，分三组，每组三个，即每组预测的boxes的数量。    
4，预测的坐标，bounding box offsets。  
1，预测的置信度。  
C，类别数。  
所以类别数为10的时候，1\*1\*[3\*(4+1+10)]=45  
![](/home/rd301/文档/学习/深度学习/目标检测/YOLOv3_cfg.png)  
#### 3.运行  
单GPU  
```
./darknet detector train cfg/voc.data cfg/yolov3-train.cfg darknet53.conv.74 | tee yolo.log   
```  
多GPU  
```
./darknet detector train cfg/voc.data cfg/yolov3-train.cfg darknet53.conv.74 -gpus 0,1,2,3 | tee yolo.log
```  
<font color=red>注意：一定要加 | tee {name}.log 这样才能保存运行过程中的各种数据，方便以后曲线可视化，比如loss跟IOU</font>  
当从某个已经保存的权重运行时：  
```
./darknet detector train cfg/coco.data cfg/yolov3.cfg backup/yolov3-train_5000.weights -gpus 0,1,2,3
```  
当突然断掉，想从某个检查点运行时：  
```
./darknet detector train cfg/coco.data cfg/yolov3.cfg backup/yolov3-loss train.backup -gpus 0,1,2,3
```    
#### 4.输出参数分析  
![](/home/rd301/文档/学习/深度学习/目标检测/YOLOv3_t.png)
Region xx:cfg文件中mask所在layer；  
Avg IOU:当前迭代中，预测的box与groundtruth box的平均交并比；  
Class：标注物体的分类准确率，越大越好，期望数值为1；  
Obj:越大越好，期望数值为1；  
No Obj:越小越好，期望数值为0；  
.5R: 当IOU的阈值为0.5的时候，recall的大小；  
0.75R:当IOU的阈值为0.75的时候，recall的大小；
7634：第几个batch；  
1.077007：总损失；  
0.980247 avg：平均损失；  
0.001000 rate: 当前的学习率；  
7.004570 seconds: 这个batch的训练时间；  
488576 image：目前为止参与训练的图片总数。  
注：输出参数跟cfg文件定义的bach,subdivision有关，比如bach=64,subdivision=16，说明一个bach处理64个样本，又分16组，每组4个样本，每组又包含3个信息 
```
Region 82 Avg IOU: 
Region 94 Avg IOU: 
Region 106 Avg IOU:
```
### 四、测试
#### 1.loss曲线跟IOU曲线  
可参考[curve_visualization.py]()
```
#coding=utf-8
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Yolov3LogVisualization:

    def __init__(self,log_path,result_dir):

        self.log_path = log_path
        self.result_dir = result_dir

    def extract_log(self, save_log_path, key_word):
        with open(self.log_path, 'r') as f:
            with open(save_log_path, 'w') as train_log:
                next_skip = False
                for line in f:
                    if next_skip:
                        next_skip = False
                        continue
                    # 去除多gpu的同步log
                    if 'Syncing' in line:
                        continue
                    # 去除除零错误的log
                    if 'nan' in line:
                        continue
                    if 'Saving weights to' in line:
                        next_skip = True
                        continue
                    if key_word in line:
                        train_log.write(line)
        f.close()
        train_log.close()

    def parse_loss_log(self,log_path, line_num=2000):
        result = pd.read_csv(log_path, skiprows=[x for x in range(line_num) if ((x % 10 != 9) | (x < 1000))],error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])
        result['loss'] = result['loss'].str.split(' ').str.get(1)
        result['avg'] = result['avg'].str.split(' ').str.get(1)
        result['rate'] = result['rate'].str.split(' ').str.get(1)
        result['seconds'] = result['seconds'].str.split(' ').str.get(1)
        result['images'] = result['images'].str.split(' ').str.get(1)

        result['loss'] = pd.to_numeric(result['loss'])
        result['avg'] = pd.to_numeric(result['avg'])
        result['rate'] = pd.to_numeric(result['rate'])
        result['seconds'] = pd.to_numeric(result['seconds'])
        result['images'] = pd.to_numeric(result['images'])
        return result

    def gene_loss_pic(self, pd_loss):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(pd_loss['avg'].values, label='avg_loss')
        ax.legend(loc='best')
        ax.set_title('The loss curves')
        ax.set_xlabel('batches')
        fig.savefig(self.result_dir + '/avg_loss')
        logger.info('save iou loss done')

    def loss_pic(self):
        train_log_loss_path = os.path.join(self.result_dir, 'train_log_loss.txt')
        self.extract_log(train_log_loss_path, 'images')
        pd_loss = self.parse_loss_log(train_log_loss_path)
        self.gene_loss_pic(pd_loss)


    def parse_iou_log(self,log_path, line_num=2000): 
        result = pd.read_csv(log_path, skiprows=[x for x in range(line_num) if (x % 10 == 0 or x % 10 == 9)],error_bad_lines=False,names=['Region Avg IOU', 'Class', 'Obj', 'No Obj', 'Avg Recall', 'count'])
        result['Region Avg IOU'] = result['Region Avg IOU'].str.split(': ').str.get(1)
        result['Class'] = result['Class'].str.split(': ').str.get(1)
        result['Obj'] = result['Obj'].str.split(': ').str.get(1)
        result['No Obj'] = result['No Obj'].str.split(': ').str.get(1)
        result['Avg Recall'] = result['Avg Recall'].str.split(': ').str.get(1)
        result['count'] = result['count'].str.split(': ').str.get(1)

        result['Region Avg IOU'] = pd.to_numeric(result['Region Avg IOU'])
        result['Class'] = pd.to_numeric(result['Class'])
        result['Obj'] = pd.to_numeric(result['Obj'])
        result['No Obj'] = pd.to_numeric(result['No Obj'])
        result['Avg Recall'] = pd.to_numeric(result['Avg Recall'])
        result['count'] = pd.to_numeric(result['count'])
        return result

    def gene_iou_pic(self, pd_loss):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(pd_loss['Region Avg IOU'].values, label='Region Avg IOU')
        # ax.plot(result['Class'].values,label='Class')
        # ax.plot(result['Obj'].values,label='Obj')
        # ax.plot(result['No Obj'].values,label='No Obj')
        # ax.plot(result['Avg Recall'].values,label='Avg Recall')
        # ax.plot(result['count'].values,label='count')
        ax.legend(loc='best')
        ax.set_title('The Region Avg IOU curves')
        ax.set_xlabel('batches')
        fig.savefig(self.result_dir + '/region_avg_iou')
        logger.info('save iou pic done')

    def iou_pic(self):
        train_log_loss_path = os.path.join(self.result_dir, 'train_log_iou.txt')
        self.extract_log(train_log_loss_path, 'IOU')
        pd_loss = self.parse_iou_log(train_log_loss_path)
        self.gene_iou_pic(pd_loss)


if __name__ == '__main__':
    log_path = '/Users/songhongwei/Downloads/nohup.log'                                 # 输出的log/txt路径
    result_dir = '/Users/songhongwei/PycharmProjects/py2project/hand/data'              # 结果储存路径
    logVis = Yolov3LogVisualization(log_path,result_dir)
    logVis.loss_pic()
    logVis.iou_pic()
```
#### 2.计算recall
```
./darknet detector recall cfg/voc.data cfg/yolov3-test.cfg results/yolov3-final.weights
```
#### 3.计算mAP  
首先将检测结果保存到result/person.txt。<font color=red>求哪一类的ap，就以类名命名。</font>
```
./darknet detector valid cfg/voc.data cfg/yolov3-test.cfg results/yolov3-final.weights -out person.txt -gpus 0,1 -thresh .5
```
下载py-faster-rcnn下的voc_eval.py
```python
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
```
新建compute_mAP.py
```python
from voc_eval import voc_eval
 
print voc_eval('./results/{}.txt', '/xxx/darknet/scripts/VOCdevkit/VOC2007/Annotations/{}.xml', '/xxx/darknet/scripts/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'person', '.')
```  
第一个参数是第一步生成的检测文件（{}.txt这个格式不要改,跟你第四个参数有关，为person.txt）；  
第二个参数是你数据集的标注文件（{}.xml这个格式不要改,跟你第五个参数有关，代表所有xml）；  
第三个参数是Main中的test.txt文件；  
第四个参数是第一步生成的文件的名字<font color=red>以自己的class命名，比如求person的ap那就是person.txt</font>；  
第五个参数是.，代表目录文件下所有文件。  
注：重复测试时，请删除./darknet/annots.pkl文件，执行python compute_mAP.py，返回的最后一个值就是AP。