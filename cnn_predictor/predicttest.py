import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from pokemon import Pokemon
# from resnet import ResNet18
# from model.predict_model import PredictModel
from model.cnnp import CNNP
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import make_interp_spline
from pylab import *

device = torch.device('cuda')
model = CNNP()
model.load_state_dict(torch.load('model\\loss=20.mdl'))#加载模型
model.eval()
dir='testcases\\124.png'
img = cv2.imread(dir)
r=img[:,:,0]
g=img[:,:,1]
b=img[:,:,2]
img_gray=np.round(r*0.299+g*0.587+b*0.114)
case1=img_gray.copy()
case2=r.copy()

tag1=[]
tag2=[]

#准备数据
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
source1 = np.expand_dims(case1, axis=0)
source2 = np.expand_dims(case2, axis=0)
source1=torch.tensor(source1).to(torch.float32)
source2=torch.tensor(source2).to(torch.float32)

#调用模型进行预测
print('loading model... ')
model.eval()
with torch.no_grad():
    predicted1 = model(source1)
    predicted2 = model(source2)

print(predicted2.shape)
predicted1  = predicted1.squeeze(dim=0)
predicted1  = predicted1.squeeze(dim=1)
predicted1  = predicted1.squeeze(dim=2)
predicted2  = predicted2.squeeze(dim=0)
predicted2  = predicted2.squeeze(dim=1)
predicted2  = predicted2.squeeze(dim=2)
predicted1 = predicted1.squeeze(1).squeeze(0)
predicted2 = predicted2.squeeze(1).squeeze(0)
predicted1 = predicted1.cpu().numpy()
predicted2 = predicted2.cpu().numpy()

predicted1=np.round(predicted1)
print(predicted1)
predicted2=np.round(predicted2)
perror=[]

for i in range(62):
    for j in range(62):
        if(((i%2)==0 and (j%2)==1) or ((i%2)==1 and (j%2)==0)):
            predicted1[i,j] = predicted2[i,j]
        if(predicted1[i,j]>255):
            predicted1[i,j] = 255
        if(predicted1[i,j]<0):
            predicted1[i,j] = 0
        error=predicted1[i,j]-r[i+1,j+1]
        if error < 5 and error > 0:
            perror.append(0)
        perror.append(error)
perror=np.array(perror)
akb=[]
ske=[]
for i in range(-255,255):
    akb.append(i)
    ske.append(np.sum(perror==i))
img[:62,:62,0]=np.array(predicted1)
# cv2.imwrite('predicted.png',img[:62,:62,:])
fig=plt.figure()
plt.suptitle('cnn predictor\'s PEs in red: ')
ax=fig.gca()


xlim(-20,+20)
plt.plot(akb,ske,color='yellow',label='cnn preditor',alpha=0.5)
plt.scatter(akb,ske,marker='*')
plt.legend(loc='upper right')
plt.savefig('cnn_red.png')
plt.show()

# print('test acc:', test_acc)
