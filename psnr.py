from random import randint
import cv2
import matplotlib.pyplot as plt
import numpy as np
from encode import encode, decode
import math
from pylab import *

#本测试需要较长时间，请耐心等待
RGB = np.array([0.299, 0.587, 0.114])

def exbit(sbit, blen, tlen):
    bits=[]
    for i in range(tlen-blen):
        bits.append(0)
    for i in sbit:
        bits.append(int(i))
    return bits

def cntzo(tags, begin, l, t):
    cnt = 0
    for i in range(begin, l):
        # print(tags[i])
        if int(tags[i]) == t:
            cnt = cnt + 1
        else:
            break
    return cnt

def compress(tags):
    l=len(tags)

    zocnts=[]
    cnt = 0
    flag = int(tags[0])
    new_tag=[flag]

    while(cnt < l):
        tlen = cntzo(tags, cnt, l, flag)
        # print(tlen, flag)
        zocnts.append(tlen)
        flag = 1 if flag==0 else 0
        cnt = cnt + tlen

    # print(zocnts)
    maxlen=max(zocnts)
    # print(maxlen)
    bit=int(np.ceil(np.log2(maxlen)))
    # print(bit)

    new_tag.extend(exbit(str(bin(bit))[2:], len(str(bin(bit)[2:])), 5))

    for tlen in zocnts:
        new_tag.extend(exbit(str(bin(int(tlen)))[2:], len(str(bin(tlen)[2:])), bit))

    # print(new_tag)
    # print(int(''.join('%s' %id for id in bits), 2))
    return new_tag
def depress(tags):
    flag=tags[0]
    temp=[]
    loc = 1
    for i in range(5):
        temp.append(tags[loc])
        loc=loc+1
    # print(temp)
    blen = int(''.join('%s' %id for id in temp), 2)
    # print(blen)
    old_tag=[]

    for i in range(int((len(tags)-6)/blen)):
        t=[]
        for j in range(blen):
            t.append(tags[i*blen+6+j])
        # print(t)
        tlen = int(''.join('%s' %id for id in t), 2)

        for j in range(tlen):
            old_tag.append(flag)
        
        flag=0 if flag==1 else 1
    
    # print(old_tag)
    return old_tag
def predictV_origin(value, grayij, X):
    beta = np.linalg.pinv(X.T * X) * X.T * value #计算矩阵的（Moore-Penrose）伪逆
    r_predict = np.linalg.det([1, grayij, grayij**2] * beta) #计算数组的行列式(beta=(xTx)^-1xTy)
    if r_predict <= min(value[1, 0], value[0, 0]): r_predict = min(value[1, 0], value[0, 0])
    elif r_predict >= max(value[1, 0], value[0, 0]):
        r_predict = max(value[1, 0], value[0, 0])
    return np.round(r_predict)


def PEs_origin(gray, img):
    pError = np.zeros(img.shape)
    predict = img.copy().astype(np.int32)
    rho = np.zeros(gray.shape)
    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            r = np.array([img[i + 1, j, 0], img[i, j + 1, 0], img[i + 1, j + 1, 0]]).reshape(3, 1)
            b = np.array([img[i + 1, j, 2], img[i, j + 1, 2], img[i + 1, j + 1, 2]]).reshape(3, 1)
            gr = np.array([gray[i + 1, j], gray[i, j + 1], gray[i + 1, j + 1]]).reshape(3, 1)
            #|1 v(i+1,j)   v(i+1,j)^2  |
            #|1 v(i,j+1)   v(i,j+1)^2  |
            #|1 v(i+1,j+1) v(i+1,j+1)^2|
            X = np.mat(np.column_stack(([1] * 3, gr, gr**2)))
            predict[i, j, 0] = predictV_origin(r, gray[i, j], X)
            predict[i, j, 2] = predictV_origin(b, gray[i, j], X)
            pError[i, j] = img[i, j] - predict[i, j]#计算e^R
            rho[i, j] = np.var([gray[i - 1, j], gray[i, j - 1], gray[i, j], gray[i + 1, j], gray[i, j + 1]], ddof=1)#方差
    return predict, pError, rho

def predictV(value, grayij, X):
    beta = np.linalg.pinv(X.T * X) * X.T * value #计算矩阵的（Moore-Penrose）伪逆
    # print("beta:  ",beta)
    r_predict = np.linalg.det([1, grayij] * beta) #计算数组的行列式(beta=(xTx)^-1xTy)
    # if r_predict <= min(value[1, 0], value[0, 0]): r_predict = min(value[1, 0], value[0, 0])
    # elif r_predict >= max(value[1, 0], value[0, 0]):
    #     r_predict = max(value[1, 0], value[0, 0])
    return np.round(r_predict)


def new_PEs(gray, img):
    pError = np.zeros(img.shape)
    predict = img.copy().astype(np.int32)
    rho = np.zeros(gray.shape)
    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            r = np.array([img[i - 1, j, 0], img[i, j + 1, 0], img[i + 1, j, 0], img[i, j - 1, 0]]).reshape(4, 1)
            b = np.array([img[i - 1, j, 2], img[i, j + 1, 2], img[i + 1, j, 2], img[i, j - 1, 2]]).reshape(4, 1)
            gr = np.array([gray[i - 1, j], gray[i, j + 1], gray[i + 1, j], gray[i, j - 1]]).reshape(4, 1)
            X = np.mat(np.column_stack(([1] * 4, gr)))
            predict[i, j, 0] = predictV(r, gray[i, j], X)
            X= np.mat(np.column_stack(([1] * 4, np.rint((gr-0.299*r)/(1-0.299)))))
            predict[i, j, 2] = predictV(b, np.round((gray[i, j]-0.299*img[i,j,0])/(1-0.299)), X)
            
            pError[i, j] = img[i, j] - predict[i, j]#计算e^R
            # print(pError[i,j,2])
            rho[i, j] = np.var([gray[i - 1, j], gray[i, j - 1], gray[i, j], gray[i + 1, j], gray[i, j + 1]], ddof=1)#方差
    return predict, pError, rho

#判断不动组
def invariant(rgb):
    return np.round(rgb[:2].dot(RGB[:2]) + 2 * (rgb[2] // 2) * RGB[2]) == np.round(rgb[:2].dot(RGB[:2]) +
                                                                                   (2 * (rgb[2] // 2) + 1) * RGB[2])#.dot算两个矩阵乘积

#判断G和G'处于何种情况,并根据不同情况进行嵌入
def JudgeG(IMG, GRAY, msg, predict, pERROR, i, msgIndex):
    c = 2
    Judge = 0
    rgbOri = IMG[i]#读取原来的rgb值
    rgbOri1 = IMG[i]
    rgbLarger = np.array([predict[i][loc] + pERROR[i][loc] for loc in range(3)]) #计算新的r、b（r'+e)
    rgbSmaller = np.array([predict[i][loc] + pERROR[i][loc] for loc in range(3)]) #计算新的r、b（r'+e)
    rgbSmaller[1] = np.floor((GRAY[i] - rgbSmaller[0] * RGB[0] - rgbSmaller[2] * RGB[2]) / RGB[1])#根据灰度的变化修改g
    rgbLarger[1] = np.ceil((GRAY[i] - rgbLarger[0] * RGB[0] - rgbLarger[2] * RGB[2]) / RGB[1])
    G1judge = np.round(rgbLarger.dot(RGB))
    G2judge = np.round(rgbSmaller.dot(RGB))
    rgbOri[1] = np.ceil((GRAY[i] - rgbOri[0] * RGB[0] - rgbOri[2] * RGB[2]) / RGB[1])#由G'逆推G（向上取整）
    rgbOri1[1] = np.floor((GRAY[i] - rgbOri1[0] * RGB[0] - rgbOri1[2] * RGB[2]) / RGB[1])#由G'逆推G（向下取整）
    G1anotherJ = np.round(rgbOri.dot(RGB))
    G2anotherJ = np.round(rgbOri1.dot(RGB))
    if G1judge == GRAY[i] and G2judge == GRAY[i]: #如果两个G'都成立
        if G1anotherJ == GRAY[i] == G2anotherJ:  #如果两个G也成立   情况四
            Judge = 4
            finalG = rgbSmaller[1] if rgbOri1[1] == IMG[i][1] else rgbLarger[1]
        else:   #只有一个G成立  情况二
            Judge = 2   
            finalG = rgbLarger[1] if msg[msgIndex+2] else rgbSmaller[1]
    else:
        if G1anotherJ == GRAY[i] == G2anotherJ:  #如果两个G也成立   情况三
            Judge = 3
            finalG = np.round((GRAY[i] - rgbSmaller[0] * RGB[0] - rgbSmaller[2] * RGB[2]))
            #c = 0 if G2judge == GRAY[i] else 1  
        else:   #只有一个G成立  情况一
            Judge = 1
            finalG = rgbLarger[1] if G1judge == GRAY[i] else rgbSmaller[1]
    
    return finalG,Judge

        


def embedMsg(img, gray, msg, mesL, selected, predict, pError, Dt):
    IMG, GRAY, pERROR = img.copy(), gray.copy(), pError.copy()
    tags = []
    La = 0
    tagsCode = '0'
    ec = 0
    location = 0
    msgIndex = 0
    # can=zip(*selected)
    xxx=[]
    black=[]
    white=[]
    for i in zip(*selected):
        if(((i[0])%2==0 and (i[1])%2 == 0) or ((i[0])%2==1 and (i[1])%2 == 1)):
            black.append(i)
        else:
            white.append(i)
    black.extend(white)
    j = 0
    SecSituation = 0
    # black=zip(*selected)
    # print(can[can%2==0])
    for i in black:  
        j += 1
        Judge = 0
        # print(i)
        if tags.count(0)*2 + SecSituation*3 < mesL:
            # 遍历满足rho<rhoT的像素点进行插入信息
            pERROR[i][0] = 2 * pERROR[i][0] + int(msg[msgIndex]) #在r层嵌入信息
            pERROR[i][2] = 2 * pERROR[i][2] + int(msg[msgIndex+1]) #在b层嵌入信息


            rgb = np.array([predict[i][loc] + pERROR[i][loc] for loc in range(3)]) #计算新的r、b（r'+e)
            x=rgb[1]
            a, Judge = JudgeG(IMG, GRAY, msg, predict, pERROR, i, msgIndex)
            #返回G' c Judge
            if Judge == 2:
                msgIndex += 1
                SecSituation += 1
            
            #else if Judge == 3
                #插入c 并且更新mesL

            D = np.linalg.norm(rgb - IMG[i]) #求d（T和T'之差的平方和开根号，相当于距离)

            if np.max(rgb) > 255 or np.min(rgb) < 0 or D > Dt:
                # print(x,rgb[1])
                tags.append(1)  # 设置当前的tag为非法（tag为1）
                # print(tags)
            else:
                tags.append(0)
                # print(i)
                # print(tags)
                msgIndex += 2
                IMG[i] = rgb
        else:
            if La == 0: #如果是第一个需嵌入的tag
                presstags=compress(tags)
                ca=len(presstags)
                # if np.unique(tags).size > 1:                   
                tagsCode, La = ''.join([str(char) for char in presstags]), len(presstags)
                    # print(tagsCode)
                # else:
                #     La = 1
            if location == La: break
            if invariant(IMG[i]): #是不动组，则把tag信息嵌入
                IMG[i][2] = 2 * (IMG[i][2] // 2) + int(tagsCode[location])
                location += 1

    if len(tags)*2 + SecSituation*3 < mesL or location < La: return False, ec, La, len(tags), tagsCode

    # print(f"=> Message: {msg}")
    # print(xxx)
    return (IMG, GRAY, pERROR), ec, La, len(tags), tagsCode

def embedMsg_origin(img, gray, msg, mesL, selected, predict, pError, Dt):
    IMG, GRAY, pERROR = img.copy(), gray.copy(), pError.copy()
    tags = []
    La = 0
    tagsCode = '0'
    ec = 0
    location = 0
    msgIndex = 0
    for i in zip(*selected):
        if tags.count(0) < mesL:
            # 遍历满足rho<rhoT的像素点进行插入信息
            pERROR[i][0] = 2 * pERROR[i][0] + int(msg[msgIndex]) #在r层嵌入信息
            pERROR[i][2] = 2 * pERROR[i][2] + ec #在b层嵌入纠错信息

            ec = abs(int(IMG[i][1] - np.round((GRAY[i] - IMG[i][0] * RGB[0] - IMG[i][2] * RGB[2]) / RGB[1])))#计算下一个纠错信息

            rgb = np.array([predict[i][loc] + pERROR[i][loc] for loc in range(3)]) #计算新的r、b（r'+e)

            rgb[1] = np.floor((GRAY[i] - rgb[0] * RGB[0] - rgb[2] * RGB[2]) / RGB[1])#根据灰度的变化修改g

            if np.round(rgb.dot(RGB)) != GRAY[i]:
                rgb[1] = np.ceil((GRAY[i] - rgb[0] * RGB[0] - rgb[2] * RGB[2]) / RGB[1])

            # rgb[1] = np.round((GRAY[i] - rgb[0] * RGB[0] - rgb[2] * RGB[2]) / RGB[1])
            if np.round(rgb.dot(RGB)) != GRAY[i]: print(f'该位置{i}无法满足灰度不变性')

            D = np.linalg.norm(rgb - IMG[i]) #求d（T和T'之差的平方和开根号，相当于距离)

            if np.max(rgb) > 255 or np.min(rgb) < 0 or D > Dt:
                tags.append(1)  # 设置当前的tag为非法（tag为1）
            else:
                tags.append(0)
                msgIndex += 1
                IMG[i] = rgb
        else:
            if La == 0: #如果是第一个需嵌入的tag
                presstags=compress(tags)
                ca=len(presstags)
                # if np.unique(tags).size > 1:                   
                tagsCode, La = ''.join([str(char) for char in presstags]), len(presstags)
                    # print(tagsCode)
                # else:
                #     La = 1
            if location == La: break
            if invariant(IMG[i]): #是不动组，则把tag信息嵌入
                IMG[i][2] = 2 * (IMG[i][2] // 2) + int(tagsCode[location])
                location += 1

    if len(tags) < mesL or location < La: return False, ec, La, len(tags), tagsCode

    # print(f"=> Message: {decode(msg)}")
    # print(xxx)
    return (IMG, GRAY, pERROR), ec, La, len(tags), tagsCode

def cvtGray(img):
    gray = np.zeros(img.shape[:-1])
    for i in np.argwhere(img[:, :, -1]):
        gray[i] = np.round(img[i].dot(RGB))
    return gray

def psnrcnt(a, rhoT, img, gray, msg, mesL, rho, predict, pError, Dt, num):
    mfig = '.'.join(target1.split('.')[:-1] + ['modified'+ str(num)]  + target1.split('.')[-1:])  # lena.modified.png
    # 根据消息长度初选 ⍴
    while np.count_nonzero(rho < rhoT) <= mesL:
        if np.count_nonzero(rho < rhoT) == rho.size:
            print('=> The picture is too small! Exit!')
            exit()
        rhoT += 1
    # 考虑参数后再选 ⍴
    enough = 0
    a=0
    rhoT = 200
    while not enough:
        # print(a)
        a=a+1
        selected = [n + 2 for n in np.where(rho[2:-2, 2:-2] < rhoT)]
        #selected = [n + 2 for n in np.where(rho[2:-2, 2:-2])]
        if selected[0].size >= (img.shape[0] - 4)**2:
            print('=> The picture is too small! Exit!')
            exit()
        enough, lastEc, La, N, tagsCode = embedMsg(img, gray, msg, mesL, selected, predict, pError, Dt)

        rhoT += 0 if enough else 1
    # print(f'=> Finish embeding msg with the critical value of p being {rhoT}')
    img, gray, pError = enough
    # 在边框中嵌入参数
    border = sorted(
        list(
            set(map(tuple, np.argwhere(gray == gray))) -
            set(map(tuple,
                    np.argwhere(gray[1:-1, 1:-1] == gray[1:-1, 1:-1]) + 1))))
    border = list(filter(lambda xy: invariant(img[xy]), border))
    if len(border) < 56:
        print('The size of image is too small to contain the necessary parameters')
        exit()

    #把rhoT、lastEc、La、N等参数转为二进制字符串，嵌入边缘的不动组中
    for char, loc in zip(f'{rhoT:016b}' + f'{lastEc:08b}' + f'{La:016b}' + f'{N:016b}',
                         filter(lambda xy: invariant(img[xy]), border)):
        img[loc][2] = 2 * (img[loc][2] // 2) + int(char)
    
    # print(f'=> Finish embeding parameters:\n\trhoT: {rhoT}, lastEc: {lastEc}, La: {La}, N: {N}')
    cv2.imwrite(mfig, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    imgRcv = cv2.imread(mfig)
    #imgRcv = cv2.cvtColor(imgRcv, cv2.COLOR_BGR2RGB)

    gt = cv2.imread('test_images\\LenaRGB.bmp')
    #img2= cv2.imread('lena.modified.png')
    x=psnr(gt,imgRcv)
    print(f'=> advanced method:Finish embeding parameters:\n\trhoT: {rhoT}, lastEc: {lastEc}, La: {La}, N: {N}, psnr: {x}')
    return x

def psnrcnt_origin(a, rhoT, img, gray, msg, mesL, rho, predict, pError, Dt, num):
    mfig_origin = '.'.join(target.split('.')[:-1] + ['modified'] + target.split('.')[-1:])
    while np.count_nonzero(rho < rhoT) <= a:
        if np.count_nonzero(rho < rhoT) == rho.size:
            print('=> The picture is too small! Exit!')
            exit()
        rhoT += 1
    # 考虑参数后再选 ⍴
    enough = 0
    while not enough:
        selected = [n + 2 for n in np.where(rho[2:-2, 2:-2] < rhoT)]
        if selected[0].size >= (img.shape[0] - 4)**2:
            print('=> The picture is too small! Exit!')
            exit()
        enough, lastEc, La, N, tagsCode = embedMsg_origin(img, gray, msg, mesL, selected, predict, pError, Dt)
        rhoT += 0 if enough else 1
    print(f'=> origin method:Finish embeding msg with the critical value of ⍴ being {rhoT}')
    
    img, gray, pError = enough
    # 在边框中嵌入参数
    border = sorted(
        list(
            set(map(tuple, np.argwhere(gray == gray))) -
            set(map(tuple,
                    np.argwhere(gray[1:-1, 1:-1] == gray[1:-1, 1:-1]) + 1))))
    border = list(filter(lambda xy: invariant(img[xy]), border))
    if len(border) < 56:
        print('The size of image is too small to contain the necessary parameters')
        exit()

    #把rhoT、lastEc、La、N等参数转为二进制字符串，嵌入边缘的不动组中
    for char, loc in zip(f'{rhoT:016b}' + f'{lastEc:08b}' + f'{La:016b}' + f'{N:016b}',
                         filter(lambda xy: invariant(img[xy]), border)):
        img[loc][2] = 2 * (img[loc][2] // 2) + int(char)
    
    
    cv2.imwrite(mfig_origin, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 读取嵌入信息的图片并计算其 predication error
    imgRcv = cv2.imread(mfig_origin)
    #imgRcv = cv2.cvtColor(imgRcv, cv2.COLOR_BGR2RGB)

    gt = cv2.imread('test_images\\LenaRGB2.bmp')
    #img2= cv2.imread('lena.modified.png')
    x=psnr(gt,imgRcv)
    print(f'=> origin method:Finish embeding parameters:\n\trhoT: {rhoT}, lastEc: {lastEc}, La: {La}, N: {N}, psnr: {x}')
    return x

def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    # 基本参数
    Size = 512
    fig = 'test_images\\LenaRGB.bmp'
    fig_origin = 'test_images\\LenaRGB2.bmp'
    target='result_images\\LenaRGB2.bmp'
    target1='result_images\\LenaRGB.bmp'
    Dt = 20
    rhoT = 0
    msg = np.random.randint(2, size=130000)
    # 读取图片
    img = cv2.imread(fig)[:Size, :Size]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cvtGray(img)

    img2 = cv2.imread(fig_origin)[:Size, :Size]
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    gray2 = cvtGray(img2)

    mfig = '.'.join(target1.split('.')[:-1] + ['modified'] + target1.split('.')[-1:])  # lena.modified.png

    mfig_origin = '.'.join(target.split('.')[:-1] + ['modified'] + target.split('.')[-1:])  # lena.modified.png
    print(f'=> Finish reading image!')

    # cv2.imwrite("TEMP.PNG", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite("TEMP_ORIGIN.PNG", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    # 计算 predict 以及 predication error
    predict, pError, rho = new_PEs(gray, img)
    predict2, pError2, rho2 = PEs_origin(gray2, img2)
    gray1=gray
    #pp=predict
    # print(pError)

    print(f'=> Finish calculating predication error!')

    #x_axis_data = [1, 3, 5, 7, 9, 11, 13]
    x_axis_data = [1, 3, 5, 7, 9, 11, 13]
    y_axis_data = [psnrcnt(10000, rhoT, img, gray, msg[:10000], 10000, rho, predict, pError, Dt, 1), 
                   psnrcnt(30000, rhoT, img, gray, msg[:30000], 30000, rho, predict, pError, Dt, 3), 
                   psnrcnt(50000, rhoT, img, gray, msg[:50000], 50000, rho, predict, pError, Dt, 5),
                   psnrcnt(70000, rhoT, img, gray, msg[:70000], 70000, rho, predict, pError, Dt, 7),
                   psnrcnt(90000, rhoT, img, gray, msg[:90000], 90000, rho, predict, pError, Dt, 9),
                   psnrcnt(110000, rhoT, img, gray, msg[:110000], 110000, rho, predict, pError, Dt, 11),
                   psnrcnt(130000, rhoT, img, gray, msg[:130000], 130000, rho, predict, pError, Dt, 13)]

    y_axis_data_origin = [psnrcnt_origin(10000, rhoT, img2, gray, msg[:10000], 10000, rho2, predict2, pError2, Dt, 1), 
                          psnrcnt_origin(30000, rhoT, img2, gray, msg[:30000], 30000, rho2, predict2, pError2, Dt, 3), 
                          psnrcnt_origin(50000, rhoT, img2, gray, msg[:50000], 50000, rho2, predict2, pError2, Dt, 5),
                          psnrcnt_origin(70000, rhoT, img2, gray, msg[:70000], 70000, rho2, predict2, pError2, Dt, 7),
                          psnrcnt_origin(90000, rhoT, img2, gray, msg[:90000], 90000, rho2, predict2, pError2, Dt, 9),
                          psnrcnt_origin(110000, rhoT, img2, gray, msg[:110000], 110000, rho2, predict2, pError2, Dt, 11),
                          psnrcnt_origin(130000, rhoT, img2, gray, msg[:130000], 130000, rho2, predict2, pError2, Dt, 13)]
    
    # 横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis_data, y_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='psnr-payloads-improve')
    plt.plot(x_axis_data, y_axis_data_origin, 'ro-', color='#28551c', alpha=0.8, linewidth=1, label='psnr-payloads')
    # 显示标签
    plt.legend(loc="upper right")
    plt.xlabel('payloads(bits) *10^4')
    plt.ylabel('PSNR(db)')

    plt.show()


    


    