from random import randint
import cv2
import matplotlib.pyplot as plt
import numpy as np
from encode import encode, decode

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
            #|1 v(i+1,j)   v(i+1,j)^2  |
            #|1 v(i,j+1)   v(i,j+1)^2  |
            #|1 v(i+1,j+1) v(i+1,j+1)^2|
            X = np.mat(np.column_stack(([1] * 4, gr)))
            predict[i, j, 0] = predictV(r, gray[i, j], X)
            X= np.mat(np.column_stack(([1] * 4, np.rint((gr-0.299*r)/(1-0.299)))))
            predict[i, j, 2] = predictV(b, np.round((gray[i, j]-0.299*img[i,j,0])/(1-0.299)), X)
            pError[i, j] = img[i, j] - predict[i, j]#计算e^R
            rho[i, j] = np.var([gray[i - 1, j], gray[i, j - 1], gray[i, j], gray[i + 1, j], gray[i, j + 1]], ddof=1)#方差
    return predict, pError, rho

#判断不动组
def invariant(rgb):
    return np.round(rgb[:2].dot(RGB[:2]) + 2 * (rgb[2] // 2) * RGB[2]) == np.round(rgb[:2].dot(RGB[:2]) +
                                                                                   (2 * (rgb[2] // 2) + 1) * RGB[2])#.dot算两个矩阵乘积

#判断G和G'处于何种情况,并根据不同情况进行嵌入
def JudgeG(IMG, GRAY, msg, predict, pERROR, i, msgIndex, percent):
    c = 2
    Judge = 0
    rgbOri = IMG#读取原来的rgb值
    rgbOri1 = IMG

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
            percent[3] += 1
            Judge = 4
            finalG = rgbSmaller[1] if rgbOri1[1] == IMG[1] else rgbLarger[1]
        else:   #只有一个G成立  情况二
            percent[1] += 1
            Judge = 2   
            finalG = rgbSmaller[1]
    elif G1judge == GRAY[i] or G2judge == GRAY[i]:
        if G1anotherJ == GRAY[i] == G2anotherJ:  #如果两个G也成立   情况三
            percent[2] += 1
            Judge = 3
            c = 0 if rgbOri1[1] == IMG[1] else 1
            finalG = rgbLarger[1] if G1judge == GRAY[i] else rgbSmaller[1]
            #c = 0 if G2judge == GRAY[i] else 1  
        else:   #只有一个G成立  情况一
            percent[0] += 1
            Judge = 1
            finalG = rgbLarger[1] if G1judge == GRAY[i] else rgbSmaller[1]
    
    return finalG, percent, Judge, c

        


def embedMsg(img, gray, msg, mesL, selected, predict, pError, Dt):
    IMG, GRAY, pERROR = img.copy(), gray.copy(), pError.copy()
    tags = []
    La = 0
    tagsCode = '0'
    ec = 0
    location = 0
    msgIndex = 0
    percent = [0,0,0,0]
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
    ThirdSituation = 0
    # black=zip(*selected)
    # print(can[can%2==0])
    for i in black:  
        c = 2
        judge = 0
        # print(i)
        if 2*tags.count(0) < mesL:
            #j += 1
            if msgIndex+1 != mesL:
                # 遍历满足rho<rhoT的像素点进行插入信息
                pERROR[i][0] = 2 * pERROR[i][0] + int(msg[msgIndex]) #在r层嵌入信息
                pERROR[i][2] = 2 * pERROR[i][2] + int(msg[msgIndex+1]) #在b层嵌入信息


                rgb = np.array([predict[i][loc] + pERROR[i][loc] for loc in range(3)]) #计算新的r、b（r'+e)
                rgb[1], percent, judge, c = JudgeG(IMG[i], GRAY, msg, predict, pERROR, i, msgIndex, percent)
                #返回G' c Judge
                
                if judge == 3:
                    ThirdSituation += 1
                    if msgIndex+2 != mesL:
                        msg = np.insert(msg, msgIndex+3, c)
                    else:
                        msg = np.insert(msg, msgIndex+2, c)
                    mesL += 1
                    #插入c 并且更新mesL

                D = np.linalg.norm(rgb - IMG[i]) #求d（T和T'之差的平方和开根号，相当于距离)

                if np.max(rgb) > 255 or np.min(rgb) < 0 or D > Dt:
                    tags.append(1)  # 设置当前的tag为非法（tag为1）
                    # print(tags)
                else:
                    tags.append(0)
                    # print(i)
                    # print(tags)
                    msgIndex += 2
                    IMG[i] = rgb
            else:   #最后只剩1bit信息
                ec = 1
                pERROR[i][0] = 2 * pERROR[i][0] + int(msg[msgIndex])
                rgb = np.array([predict[i][loc] + pERROR[i][loc] for loc in range(3)]) #计算新的r、b（r'+e)
                rgb[1] = np.floor((GRAY[i] - rgb[0] * RGB[0] - rgb[2] * RGB[2]) / RGB[1])
                if np.round(rgb.dot(RGB)) != GRAY[i]:
                    rgb[1] = np.ceil((GRAY[i] - rgb[0] * RGB[0] - rgb[2] * RGB[2]) / RGB[1])
                D = np.linalg.norm(rgb - IMG[i])
                if np.max(rgb) > 255 or np.min(rgb) < 0 or D > Dt:
                    tags.append(1)  # 设置当前的tag为非法（tag为1）
                else:
                    tags.append(0)
                    msgIndex += 1
                    IMG[i] = rgb
        else:
            if La == 0: #如果是第一个需嵌入的tag
                #presstags=compress(tags)
                
                ca=len(tags)
                # if np.unique(tags).size > 1:                   
                tagsCode, La = ''.join([str(char) for char in tags]), len(tags)
                    # print(tagsCode)
                # else:
                #     La = 1
            if location == La: break
            if invariant(IMG[i]): #是不动组，则把tag信息嵌入
                IMG[i][2] = 2 * (IMG[i][2] // 2) + int(tagsCode[location])
                location += 1

    if 2*tags.count(0) < mesL or location < La: return False, ec, La, len(tags), tagsCode
    #print(j)
    # print(percent)
    # print(percent[0]/(percent[0]+percent[1]+percent[2]+percent[3]))
    # print(percent[1]/(percent[0]+percent[1]+percent[2]+percent[3]))
    # print(percent[2]/(percent[0]+percent[1]+percent[2]+percent[3]))
    # print(percent[3]/(percent[0]+percent[1]+percent[2]+percent[3]))
    #print(f"=> Message: {msg}")
    # print(xxx)
    return (IMG, GRAY, pERROR), ec, La, len(tags), tagsCode


def cvtGray(img):
    gray = np.zeros(img.shape[:-1])
    for i in np.argwhere(img[:, :, -1]):
        gray[i] = np.round(img[i].dot(RGB))
    return gray

#判断被修改的像素是哪种情况并计算正确的G
def JudgeAno(IMG,IMGchange, GRAY, predict, pError, i, c):
    Judge = 0
    rgbOri = IMG #恢复rb
    rgbOri1 = IMG #恢复rb
    rgbOri[1] = np.ceil((GRAY[i] - rgbOri[0] * RGB[0] - rgbOri[2] * RGB[2]) / RGB[1])#计算G（向上取整）
    rgbOri1[1] = np.floor((GRAY[i] - rgbOri1[0] * RGB[0] - rgbOri1[2] * RGB[2]) / RGB[1])#G（向下取整） 

    rgbLarger = IMGchange#读取修改后的rgb值
    rgbSmaller = IMGchange
    rgbSmaller[1] = np.floor((GRAY[i] - rgbSmaller[0] * RGB[0] - rgbSmaller[2] * RGB[2]) / RGB[1])#计算G'（向下）
    rgbLarger[1] = np.ceil((GRAY[i] - rgbLarger[0] * RGB[0] - rgbLarger[2] * RGB[2]) / RGB[1])#计算G'（向上）

    G1judge = np.round(rgbLarger.dot(RGB))#G'是否符合条件
    G2judge = np.round(rgbSmaller.dot(RGB))

    G1anotherJ = np.round(rgbOri.dot(RGB))#G是否符合条件
    G2anotherJ = np.round(rgbOri1.dot(RGB))
    if c != 2:  #若像素点是第三种情况的第二个像素点
        finalG = rgbOri[0] if c == 0 else rgbOri[1]
        return finalG, Judge
    if G1judge == GRAY[i] and G2judge == GRAY[i]: #如果两个G'都成立
        if G1anotherJ == GRAY[i] == G2anotherJ:  #如果两个G也成立   情况四
            Judge = 4
            finalG = rgbOri1[1] if rgbSmaller[1] == IMG[1] else rgbOri[1]
        else:   #只有一个G成立  情况二
            Judge = 2   
            finalG = rgbOri[1] if G1anotherJ == GRAY[i] else rgbOri1[1]
    else:
        if G1anotherJ == GRAY[i] == G2anotherJ:  #如果两个G也成立   情况三
            Judge = 3
            finalG = np.round((GRAY[i] - rgbOri[0] * RGB[0] - rgbOri[2] * RGB[2]))
            #c = 0 if G2judge == GRAY[i] else 1  
        else:   #只有一个G成立  情况一
            Judge = 1
            finalG = rgbOri[1] if G1anotherJ == GRAY[i] else rgbOri1[1]
    
    return finalG,Judge

if __name__ == '__main__':
    # 基本参数
    Size = 60
    fig = 'test_images\\LenaRGB.bmp'
    Dt = 20
    rhoT = 0
   
    msg = [1,1,1,1,0,0,0,0, 1,1,1,1,0,0,0,0, 1,1,1,1,0,0,0,0, 1,1,1,1,0,0,0,0]
    mesL = 32
    print("mesL:", mesL)
    print("msg:", msg)
    #print(encode(msg))
    # 读取图片
    img = cv2.imread(fig)[:Size, :Size]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cvtGray(img)
    mfig = '.'.join(fig.split('.')[:-1] + ['modified'] + fig.split('.')[-1:])  # lena.modified.png
    #print(f'{img}\n=> Finish reading image!')
    # 准备嵌入前后灰度对比图
    # plt.figure(figsize=(12, 6)), plt.suptitle('Grayscale')
    # plt.subplot(1, 2, 1), plt.title('Origin')
    # plt.hist(gray.ravel(), 256)
    # plt.show(block=False)

    #cv2.imwrite("TEMP.PNG", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # 计算 predict 以及 predication error
    predict, pError, rho = new_PEs(gray, img)
    gray1=gray
    pp=predict
    # print(pError)

    # plt.figure(figsize=(12, 6)), plt.suptitle('error hist')
    # plt.hist(pError.ravel())

    print(f'=> Finish calculating predication error!')
    # 根据消息长度初选 ⍴
    while np.count_nonzero(rho < rhoT) <= mesL:
        if np.count_nonzero(rho < rhoT) == rho.size:
            print('=> The picture is too small! Exit!')
            exit()
        rhoT += 1
    # 考虑参数后再选 ⍴
    enough = 0
    a=0
    #rhoT = 100
    while not enough:
        #print(a)
        a=a+1
        selected = [n + 2 for n in np.where(rho[2:-2, 2:-2] < rhoT)]
        if selected[0].size >= (img.shape[0] - 4)**2:
            print('=> The picture is too small! Exit!')
            exit()
        enough, lastEc, La, N, tagsCode = embedMsg(img, gray, msg, mesL, selected, predict, pError, Dt)

        rhoT += 0 if enough else 1
    Sign = lastEc
    print(f'=> Finish embeding msg with the critical value of p being {rhoT}')
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
    #print(f'=> Finish embeding parameters:\n\trhoT: MessageLen: {mesL}, N: {N}')



    print(f'=> Finish embeding parameters:\n\trhoT: {rhoT}, lastEc: {lastEc}, La: {La}, N: {N}, tagsCode: {tagsCode}')
    cv2.imwrite(mfig, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 读取嵌入信息的图片并计算其 predication error
    imgRcv = cv2.imread(mfig)
    imgRcv = cv2.cvtColor(imgRcv, cv2.COLOR_BGR2RGB)
    grayRcv = cvtGray(imgRcv)
    predictRcv, pErrorRcv, rhoRcv = new_PEs(grayRcv, imgRcv)#重新计算pe值

    print(f'=> Finish reading embeded image and calculating predication error!')

    #画直方图验证灰度不变性
    # plt.subplot(1, 2, 2), plt.title('Modified')
    # plt.hist(grayRcv.ravel(), 256)
    # plt.show(block=False)
    print(f'=> Ensure the grayscale invariant: {np.all(gray == grayRcv)}')

    #提取边框的参数
    border = sorted(
        list(
            set(map(tuple, np.argwhere(grayRcv == grayRcv))) -
            set(map(tuple,
                    np.argwhere(grayRcv[1:-1, 1:-1] == grayRcv[1:-1, 1:-1]) + 1))))
    border = [str(imgRcv[loc][2] % 2) for loc in filter(lambda xy: invariant(imgRcv[xy]), border)]
    rhoT = int(''.join(border[:16]), 2)
    lastEc = int(''.join(border[16:24]), 2)
    La = int(''.join(border[24:40]), 2)
    N = int(''.join(border[40:56]), 2)

    # selected = [tuple(n + 2) for n in np.argwhere(rhoRcv[2:-2, 2:-2] < rhoT)]#重新选出方差符合的块
    selected1 = [n + 2 for n in np.where(rhoRcv[2:-2, 2:-2] < rhoT)]
    black=[]
    white=[]
    for i in zip(*selected1):
        # print(i)
        if(((i[0])%2==0 and (i[1])%2 == 0) or ((i[0])%2==1 and (i[1])%2 == 1)):
            black.append(i)
        else:
            white.append(i)
    black.extend(white)

    tagsCode1=[]

    for i in black[N:]:
        # print(i)
        if invariant(imgRcv[i]):
            # print(i)
            tagsCode1.append(imgRcv[i][2] % 2)

    # print(tagsCode1)
    nsgRcv = msg
    candidate=[]
    count = 0
    x=0
    for i in range(N):
        # print(black[i])
        if tagsCode1[i] == 0:
            count += 1
            candidate.append(black[i])
            # print(black[i])
    # print(count)
    candidate.reverse() 
   
    print(
        f'=> Finish extracting parameters:\n\trhoT: {rhoT}, lastEc: {lastEc}, La: {La}, N: {N}'
    )

    # 根据参数去提取嵌入的信息
    # candidate = reversed([selected[:N][index] for index, value in enumerate(tagsCode) if value == 0])#逆序遍历找到tag为0的像素
    predictRcv = imgRcv.copy().astype(np.int32)
    pErrorRcv = np.zeros(imgRcv.shape)
    msgRcv = []
    count1 = 0
    judge = 0
    num = 0
    # print(np.array_equal(tagsCode,tagsCode1))
    # #predictRcv, pErrorRcv, rhoRcv = new_PEs(grayRcv, imgRcv)#重新计算pe值
    # print(np.array_equal(predictRcv,predict))
    for i in candidate:
        count1 += 1
        # print(i)
        rM = np.array([imgRcv[i[0] - 1, i[1], 0], imgRcv[i[0], i[1] + 1, 0],
                       imgRcv[i[0] + 1, i[1], 0], imgRcv[i[0], i[1] - 1, 0]]).reshape(4, 1)
        bM = np.array([imgRcv[i[0] - 1, i[1], 2], imgRcv[i[0], i[1] + 1, 2],
                       imgRcv[i[0] + 1, i[1], 2], imgRcv[i[0], i[1] - 1, 2]]).reshape(4, 1)
        grM = np.array([grayRcv[i[0] - 1, i[1]], grayRcv[i[0], i[1] + 1], 
                        grayRcv[i[0] + 1, i[1]], grayRcv[i[0], i[1] - 1]]).reshape(4, 1)
        # grM = np.array([gray1[i[0] - 1, i[1]], gray1[i[0], i[1] + 1], 
        #                 gray1[i[0] + 1, i[1]], gray1[i[0], i[1] - 1]]).reshape(4, 1)
        X = np.mat(np.column_stack(([1] * 4, grM)))

        predictRcv[i][0] = predictV(rM, grayRcv[i], X)

        X= np.mat(np.column_stack(([1] * 4, np.rint(grM-0.299*rM))))
        predictRcv[i][2] = predictV(bM, grayRcv[i], X) 
        # print(pError)
        pErrorRcv[i] = imgRcv[i] - predictRcv[i]#这里为什么要重新算pe？

        imgChange = imgRcv[i]
        imgRcv[i] = predictRcv[i] + pErrorRcv[i] // 2
        imgRcv[i][1], judge = JudgeAno(imgRcv[i],imgChange, grayRcv, predictRcv, pErrorRcv, i, 2)
        if Sign == 0 or count1 != 1:
            msgRcv.append(int(pErrorRcv[i][2]) % 2) #提取消息
            msgRcv.append(int(pErrorRcv[i][0]) % 2) #提取消息
            if count1 == 1:
                pErrorRcv[i] = pErrorRcv[i] // 2 #恢复perror
                imgRcv[i] = predictRcv[i] + pErrorRcv[i]
                imgRcv[i][1], judge = JudgeAno(imgRcv[i], grayRcv, predictRcv, pErrorRcv, i, 2)
                num += 2
            elif judge == 3 and (count1 != 2 or Sign == 0):
                msgRcv[num-2] = msgRcv[num-1]
                msgRcv[num-1] = msgRcv[num]
                msgRcv[num] = msgRcv[num+1]
                msgRcv.pop()
                pErrorRcv[i] = pErrorRcv[i] // 2 #恢复perror
                imgRcv[i] = predictRcv[i] + pErrorRcv[i]
                imgRcv[i][1], judge = JudgeAno(imgRcv[i], grayRcv, predictRcv, pErrorRcv, i, int(pErrorRcv[lasti][2]) % 2)
                num += 1
            elif judge == 3 and count1 == 2:
                msgRcv[num-1] = msgRcv[num]
                msgRcv[num] = msgRcv[num+1]
                msgRcv.pop()
                pErrorRcv[i] = pErrorRcv[i] // 2 #恢复perror
                imgRcv[i] = predictRcv[i] + pErrorRcv[i]
                imgRcv[i][1], judge = JudgeAno(imgRcv[i], grayRcv, predictRcv, pErrorRcv, i, int(pErrorRcv[lasti][2]) % 2)
                num += 1
            else:
                pErrorRcv[i] = pErrorRcv[i] // 2 #恢复perror
                imgRcv[i] = predictRcv[i] + pErrorRcv[i]
                imgRcv[i][1], judge = JudgeAno(imgRcv[i], imgChange, grayRcv, predictRcv, pErrorRcv, i, 2)
                num += 2
        elif count1 == 1:
            msgRcv.append(int(pErrorRcv[i][0]) % 2) #提取消息
            num += 1

        nextEc = pErrorRcv[i][2] % 2 #下一个纠错位
        judge = 0
        lasti = i

    print("received msg:", nsgRcv)
    print(f"=> The msg is equal to received msg: {msg == nsgRcv}")
    # plt.savefig('Grayscale.png')
    # plt.show()

