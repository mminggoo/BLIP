import os
import json
import pandas as pd

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

def csv2anno(df, image_root, file_path="annotation/coyo_val.json"):
    datas = {'annotations': [], 'images': []}
    for i in range(len(df)):
        ann = {'image_id': i, 'caption': df.iloc[i]['text'], 'id': i}
        images = {'id': i}
        
        datas['annotations'].append(ann)
        datas['images'].append(images)

    with open(file_path, 'w') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)


class coco_karpathy_train(Dataset):
    def __init__(self, transform, image_root, df_root, max_words=30, prompt=''):        
        
        self.transform = transform
        self.df = pd.read_csv(df_root).iloc[:40]
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):    
        
        ann = self.df.iloc[index]
        
        image_path = os.path.join(self.image_root, ann['image_file'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['text'], self.max_words) 

        return image, caption, index
    
    
class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, df_root, split):
        
        self.df = pd.read_csv(df_root).iloc[:20]
        self.transform = transform
        self.image_root = image_root
        csv2anno(self.df, self.image_root, file_path='annotation/coyo_val.json')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):    
        
        ann = self.df.iloc[index]
        
        image_path = os.path.join(self.image_root, ann['image_file'])         
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)          
        
        return image, index  
    
class coco_karpathy_caption_eval2(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)   

    
class coco_karpathy_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
