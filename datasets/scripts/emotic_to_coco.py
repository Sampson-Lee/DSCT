# -*- coding: utf-8 -*-
from ctypes.wintypes import LANGID
import os, cv2, json, csv
from sys import prefix
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
import torchfile
from IPython import embed
from tqdm import tqdm

category_list = ['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence','Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement','Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity','Suffering','Surprise','Sympathy','Yearning']

class MyData2COCO:
    def __init__(self):
        self.images = []        
        self.annotations = []   
        self.categories = []   
        self.img_id = 0         
        self.ann_id = 0         
 
    def _categories(self):               
        for i in range(1, len(category_list)+1):
            category = {}
            category['id'] = i
            category['name'] = str(category_list[i-1])          
            category['supercategory'] = 'expression'    
            self.categories.append(category)

    def _image(self, path, h, w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = path #os.path.basename(path)
        return image

    def _annotation(self, label, bbox):
        area = bbox[2] * bbox[3]
        points = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0], bbox[1] + bbox[3]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(''.join([str(x) for x in label]), 2)
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = bbox
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def to_coco(self, csv_file, img_folder):
        annotations = torchfile.load(csv_file)
        # embed()
        last_img = ''
        for item in tqdm(annotations):
            img_name = str(item[b'filename'], encoding='utf-8')
            folder = str(item[b'folder'], encoding='utf-8')
            img_name = os.path.join(folder, img_name)

            bbox_list = list(item[b'head_bbox'])
            boxes[0::2].clamp_(min=0, max=w)
            boxes[1::2].clamp_(min=0, max=h)
            
            abs_bb_w = bbox_list[2]-bbox_list[0]
            abs_bb_h = bbox_list[3]-bbox_list[1]
            if abs_bb_h<=0 or abs_bb_w<=0:
                print('invalid bbox')
                continue
            abs_bb_left_x = bbox_list[0]
            abs_bb_top_y = bbox_list[1]

            workers = item[b'workers']
            disc_labels = np.zeros(len(category_list), dtype = int)
            for worker in workers:
                for cate in worker[b'labels']:
                    disc_labels[cate-1] = disc_labels[cate-1] + 1
            disc_labels = np.clip(disc_labels, 0, 1)

            if img_name == last_img:
                # print('same image', img_name)
                annotation = self._annotation(disc_labels, [abs_bb_left_x,abs_bb_top_y,abs_bb_w,abs_bb_h])
                self.annotations.append(annotation)
                self.ann_id += 1 # start fron 0
            else: 
                img_path = os.path.join(img_folder, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    # embed()
                    print(img_path)
                    continue
                h, w, c = img.shape

                self.img_id += 1 #start fron 0,but set -1 at the beginning
                self.images.append(self._image(img_name, h, w))
                annotation = self._annotation(disc_labels, [abs_bb_left_x,abs_bb_top_y,abs_bb_w,abs_bb_h])
                self.annotations.append(annotation)
                self.ann_id += 1 #start fron 0
                last_img = img_name

        instance = {}
        instance['info'] = 'EMOTIC-Detection'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def save_coco_json(self, instance, save_path):
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent = 4, separators = (',', ': '))

if __name__ == '__main__':
    FACE_to_COCO = MyData2COCO()
    file_type = 'train'
    data_path = '/home/xinpeng/EMOTIC/images/'
    csv_file = '/home/xinpeng/EMOTIC/annotations/DiscreteContinuousAnnotations26_' + file_type+'.t7'
    output_file = '/home/xinpeng/EMOTIC/annotations/' + file_type+'_bi.json'
    train_instance = FACE_to_COCO.to_coco(csv_file, data_path)
    FACE_to_COCO.save_coco_json(train_instance, output_file)

    dataset = torchvision.datasets.CocoDetection(data_path, output_file)
    # embed()
    for img, target in dataset:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for ann in target:
            bbox = list(map(int, ann['bbox']))
            bbox = np.array([
                [bbox[0], bbox[1]],
                [bbox[0]+ bbox[2], bbox[1]],
                [bbox[0]+ bbox[2], bbox[1]+bbox[3]],
                [bbox[0], bbox[1]+ bbox[3]],
                ])
            bbox = bbox.reshape((4, 2))
            cv2.polylines(img, [bbox], True, (0, 255, 0), 2)
            class_bi = bin(ann["category_id"])[2:]
            class_bi = np.array([0,]*(len(category_list) - len(class_bi)) + [int(x) for x in class_bi])
            
            class_idx = np.where(class_bi==1)[0]
            label_str = ''
            for i in class_idx:
                label_str = label_str + ' ' + category_list[i]
            cv2.putText(img, label_str, (bbox[3][0], bbox[3][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
        cv2.imwrite("test.jpg", img)
        # exit()