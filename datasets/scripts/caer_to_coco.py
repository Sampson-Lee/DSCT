# from ctypes.wintypes import LANGID
import os,cv2,tqdm,json,csv
from sys import prefix
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision

category_list = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class MyData2COCO:
    def __init__(self):
        self.images = []        
        self.annotations = []   
        self.categories = []    
        self.img_id = 0        
        self.ann_id = 0        
 
    def _categories(self):             
        for i in range(len(category_list)):
            category = {}
            category['id'] = i
            category['name'] = str(category_list[i])            
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
        annotation['category_id'] = label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = bbox
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def to_coco(self, input_file, img_folder):
        category_list = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        with open(input_file, newline = '') as csvfile:
            spamreader = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
            #next(spamreader)#skip the first line
            last_img = ''
            self._categories()  # 初始化categories基本信息
            num = 0
            for single_data in spamreader:
                # if num==200: break
                num = num + 1
                single_data = single_data[0].split(',')
                img_name = single_data[0]
                class_index = int(single_data[1])
                bbox_list = list(map(float, single_data[2:]))

                abs_bb_w = bbox_list[2]-bbox_list[0]
                abs_bb_h = bbox_list[3]-bbox_list[1]
                abs_bb_left_x = bbox_list[0]
                abs_bb_top_y = bbox_list[1]

                if img_name == last_img:
                    annotation = self._annotation(class_index, [abs_bb_left_x,abs_bb_top_y,abs_bb_w,abs_bb_h])
                    self.annotations.append(annotation)
                    self.ann_id += 1 #start fron 0

                else:
                    img_path = os.path.join(img_folder, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(img_path)
                        continue
                    h, w, c = img.shape

                    self.img_id += 1 #start fron 0,but set -1 at the beginning
                    self.images.append(self._image(img_name, h, w))
                    annotation = self._annotation(class_index, [abs_bb_left_x,abs_bb_top_y,abs_bb_w,abs_bb_h])
                    self.annotations.append(annotation)
                    self.ann_id += 1 #start fron 0
                    last_img = img_name

        instance = {}
        instance['info'] = 'CAER'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def save_coco_json(self, instance, save_path):
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent = 4, separators = (',', ': '))

if __name__ == '__main__':
    from IPython import embed
    FACE_to_COCO = MyData2COCO()
    file_type = 'train'
    data_path = '/data1/xinpeng/CAER_S/' + file_type
    input_file = '/data1/xinpeng/CAER_S/' + file_type+'.csv'
    output_file = '/data1/xinpeng/CAER_S/' + file_type+'.json'
    # train_instance = FACE_to_COCO.to_coco(input_file, data_path)
    # FACE_to_COCO.save_coco_json(train_instance, output_file)

    dataset = torchvision.datasets.CocoDetection(data_path, output_file)
    for img, target in dataset:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for ann in target:
            class_idx = ann["category_id"]
            print(dataset.coco.imgs[target[0]["image_id"]]['file_name'])
            box = ann["bbox"]
            bbox = list(map(int,box))
            bbox = np.array([
                [bbox[0], bbox[1]],
                [bbox[0]+ bbox[2], bbox[1]],
                [bbox[0]+ bbox[2], bbox[1]+bbox[3]],
                [bbox[0], bbox[1]+ bbox[3]],
                ])
            bbox = bbox.reshape((4, 2))
            cv2.polylines(img, [bbox], True, (0, 255, 0), 2)
            cv2.putText(img, category_list[class_idx], (bbox[3][0], bbox[3][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            
        cv2.imwrite("test.jpg", img)
        # exit()