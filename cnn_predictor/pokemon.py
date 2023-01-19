import torch
import os, glob
# import random, csv
from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
import cv2
import numpy as np

class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        self.images=[]
        self.labels=[]

        files = os.listdir(root)
        tags=[]
        imgs=[]
        
        # 3
        for name in files[:self.resize]:
            size=64
            # print(name)
            img = cv2.imread(root+name)
            r=img[:,:,0]
            g=img[:,:,1]
            b=img[:,:,2]
            img_gray=np.round(r*0.299+g*0.587+b*0.114)
            case1=img_gray.copy()
            case2=r.copy()


            # print(img_gray[0:4,:4])
            # print(r[0:4,:4])
            # print(g[0,:16])
            # print(b[0,:16])
            tag1=[]
            tag2=[]
            
            for i in range(64):
                for j in range(64):
                    if(((i%2)==0 and (j%2)==0) or ((i%2)==1 and (j%2)==1)):
                        if(i>0 and j>0 and i<63 and j<63):
                            tag2.append(r[i,j])
                            tag2.append(r[i,j])
                        case1[i,j] = r[i,j]
                        case2[i,j] = img_gray[i,j]
                    if(((i%2)==0 and (j%2)==1) or ((i%2)==1 and (j%2)==0)):
                        if(i>0 and j>0 and i<63 and j<63):
                            tag1.append(r[i,j])
                            tag1.append(r[i,j])
            case1 = np.expand_dims(case1, axis=0)
            case2 = np.expand_dims(case2, axis=0)
            # print(case1.shape)
            self.images.append(case1)
            self.images.append(case2)
            self.labels.append(tag1)
            self.labels.append(tag2)
            # print(case1[:4,:4])            
            # print(case2[:4,:4])   
        # print(len(self.labels))
            # print(tag)
            # if k ==9:    
            #     print(tags)
            #     test = torch.tensor(tags)
            #     variabley = torch.Tensor(imgs) 
            #     # print(imgs)  
            #     print(variabley)  
            #     print(test)     
            #     # print(type(variabley))  
            #     break
            # k=k+1
        # for name in sorted(os.listdir(root)):
        #     if not os.path.isdir(os.path.join(root, name)):
        #         continue
        #     self.name2label[name] = len(self.name2label.keys())
        # self.images, self.labels = self.load_csv('images')
        # print(self.images)


        # print(len(self.images))
        if mode == 'train':
            self.images = self.images[:int(0.8 * len(self.images))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.8 * len(self.labels)):int(0.9*len(self.images))]
            self.labels = self.labels[int(0.8 * len(self.labels)):int(0.9*len(self.labels))]
        else:
            self.images = self.images[int(0.9 * len(self.images)):]
            self.labels = self.labels[int(0.9 * len(self.labels)):]
        
        
    # def load_csv(self, filename):
    #     if not os.path.exists(os.path.join(self.root, filename)):
    #         images = []
    #         for name in self.name2label.keys():
    #             images += glob.glob(os.path.join(self.root, name, '*.png'))
    #             # images += glob.glob(os.path.join(self.root, name, '*.jpg'))
    #             # images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
    #             # images += glob.glob(os.path.join(self.root, name, '*.gif'))
    #         print(len(images), images)
    #         random.shuffle(images)
    #         with open(os.path.join(self.root, filename), mode='w', newline='') as f:
    #             writer = csv.writer(f)
    #             for img in images:
    #                 name = img.split(os.sep)[-2]
    #                 label = self.name2label[name]
    #                 writer.writerow([img, label])
    #             print('written into csv file:', filename)
    #     images, labels = [], []
    #     with open(os.path.join(self.root, filename)) as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             img, label = row
    #             label = int(label)
    #             images.append(img)
    #             labels.append(label)
    #     assert len(images) == len(labels)
    #     return images, labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, key):
        # print(len(self.labels))
       
        # print(self.images)
        img, label = self.images[key], self.labels[key]
        # trans = transforms.Compose([
        #     lambda x: Image.open(x).convert('RGB'),
        #     transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
        #     transforms.RandomRotation(15),
        #     transforms.CenterCrop(self.resize),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        # img = trans(img)
        img=torch.tensor(img)
        label = torch.tensor(label)
        # print(label.size())
        img = img.unsqueeze(1)
        # print(1111111111111111111)
        # print(img,label)
        return img, label
