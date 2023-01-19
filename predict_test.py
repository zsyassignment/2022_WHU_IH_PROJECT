from random import randint
import cv2
import matplotlib.pyplot as plt
import numpy as np
from encode import encode, decode
import seaborn as sns
from pylab import *
from scipy.interpolate import make_interp_spline

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

def predictV_old(value, grayij, X):
    beta = np.linalg.pinv(X.T * X) * X.T * value #计算矩阵的（Moore-Penrose）伪逆
    r_predict = np.linalg.det([1, grayij, grayij**2] * beta) #计算数组的行列式(beta=(xTx)^-1xTy)
    if r_predict <= min(value[1, 0], value[0, 0]): r_predict = min(value[1, 0], value[0, 0])
    elif r_predict >= max(value[1, 0], value[0, 0]):
        r_predict = max(value[1, 0], value[0, 0])
    return np.round(r_predict)


def PEs(gray, img):
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
            predict[i, j, 0] = predictV_old(r, gray[i, j], X)
            predict[i, j, 2] = predictV_old(b, gray[i, j], X)
            pError[i, j] = img[i, j] - predict[i, j]#计算e^R
            rho[i, j] = np.var([gray[i - 1, j], gray[i, j - 1], gray[i, j], gray[i + 1, j], gray[i, j + 1]], ddof=1)#方差
    return predict, pError, rho

def predictV(value, grayij, X):
    beta = np.linalg.pinv(X.T * X) * X.T * value #计算矩阵的（Moore-Penrose）伪逆
    # print("beta:  ",beta)
    r_predict = np.linalg.det([1, grayij] * beta) #计算数组的行列式(beta=(xTx)^-1xTy)
    # print('before:',np.round(r_predict))
    # if r_predict <= min(value[1, 0], value[0, 0]): r_predict = min(value[1, 0], value[0, 0])
    # elif r_predict >= max(value[1, 0], value[0, 0]):
    #     r_predict = max(value[1, 0], value[0, 0])
    # print('after:',np.round(r_predict))
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
            # print('X: ',X)
            X= np.mat(np.column_stack(([1] * 4, np.rint((gr-0.299*r)/(1-0.299)))))
            # print('gr: ', gr)
            # print('r: ', r)
            # print('modified X: ',X)
            # print()
            predict[i, j, 2] = predictV(b, np.round((gray[i, j]-0.299*img[i,j,0])/(1-0.299)), X)
            
            pError[i, j] = img[i, j] - predict[i, j]#计算e^R
            # print(pError[i,j,2])
            rho[i, j] = np.var([gray[i - 1, j], gray[i, j - 1], gray[i, j], gray[i + 1, j], gray[i, j + 1]], ddof=1)#方差
    return predict, pError, rho

#判断不动组
def invariant(rgb):
    return np.round(rgb[:2].dot(RGB[:2]) + 2 * (rgb[2] // 2) * RGB[2]) == np.round(rgb[:2].dot(RGB[:2]) +
                                                                                   (2 * (rgb[2] // 2) + 1) * RGB[2])#.dot算两个矩阵乘积
    
def embedMsg(img, gray, msg, mesL, selected, predict, pError, Dt):
    a=0
    IMG, GRAY, pERROR = img.copy(), gray.copy(), pError.copy()
    tags = []
    La = 0
    tagsCode = '0'
    ec = 0
    location = 0
    msgIndex = 0
    # can=zip(*selected)
    xxx=[]
    # black=[]
    # white=[]
    # for i in zip(*selected):
    #     if(((i[0])%2==0 and (i[1])%2 == 0) or ((i[0])%2==1 and (i[1])%2 == 1)):
    #         black.append(i)
    #     else:
    #         white.append(i)
    # black.extend(white)
    # # black=zip(*selected)
    # # print(can[can%2==0])

    black=[]
    white=[]
    for i in zip(*selected):
        if(((i[0])%2==0 and (i[1])%2 == 0) or ((i[0])%2==1 and (i[1])%2 == 1)):
            black.append(i)
        else:
            white.append(i)
    black.extend(white)
    # print(black)

    brho=[]
    order=[]
    for i in black:
        brho.append(rho[i])
        order.append(i)
    zipped = zip(order,brho)
    sort_zipped = sorted(zipped,key=lambda x:(x[1],x[0]))
    result = zip(*sort_zipped)
    x_axis, y_axis = [list(x) for x in result]

    for i in x_axis:  
        # print(i)
        if tags.count(0) < mesL:
            # 遍历满足rho<rhoT的像素点进行插入信息
            pERROR[i][0] = 2 * pERROR[i][0] + int(msg[msgIndex]) #在r层嵌入信息
            pERROR[i][2] = 2 * pERROR[i][2] + ec #在b层嵌入纠错信息

            ec = abs(int(IMG[i][1] - np.round((GRAY[i] - IMG[i][0] * RGB[0] - IMG[i][2] * RGB[2]) / RGB[1])))#计算下一个纠错信息
            xxx.append(ec)
            rgb = np.array([predict[i][loc] + pERROR[i][loc] for loc in range(3)]) #计算新的r、b（r'+e)

            rgb[1] = np.floor((GRAY[i] - rgb[0] * RGB[0] - rgb[2] * RGB[2]) / RGB[1])#根据灰度的变化修改g

            if np.round(rgb.dot(RGB)) != GRAY[i]:
                rgb[1] = np.ceil((GRAY[i] - rgb[0] * RGB[0] - rgb[2] * RGB[2]) / RGB[1])

            # rgb[1] = np.round((GRAY[i] - rgb[0] * RGB[0] - rgb[2] * RGB[2]) / RGB[1])
            if np.round(rgb.dot(RGB)) != GRAY[i]: print(f'该位置{i}无法满足灰度不变性')

            D = np.linalg.norm(rgb - IMG[i]) #求d（T和T'之差的平方和开根号，相当于距离)

            if np.max(rgb) > 255 or np.min(rgb) < 0 or D > Dt:#
                tags.append(1)  # 设置当前的tag为非法（tag为1）
                # print(tags)
                a=a+1
            else:
                tags.append(0)
                # print(i)
                # print(tags)
                a=a+1
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

def test(Size, name,fig, histdir,histdir_blue, predir):

    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    matplotlib.rcParams['axes.unicode_minus'] =False

    # 基本参数
    Dt = 20
    rhoT = 0
    msg = '314159265659314159265659'
    mesL = len(encode(msg))

    # 读取图片
    img = cv2.imread(fig)[:Size, :Size]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cvtGray(img)
    mfig = '.'.join(fig.split('.')[:-1] + ['modified'] + fig.split('.')[-1:])  # lena.modified.png
    print(f'=> Finish reading image!')

    # cv2.imwrite("TEMP.PNG", cv2.cvtColor(gray, cv2.COLOR_RGB2BGR))
    # 计算 predict 以及 predication error
    predict, pError, rho = new_PEs(gray, img)
    predict, pError1, rho = PEs(gray, img)

    # sns.distplot(pError[:,:,2],hist=False,color='r',label='diamond') #中图
    # sns.distplot(pError1[:,:,2],hist=False,color='b',label='original') #中图

    ap=pError[:,:,0]
    ap=np.array(ap)
    # print(np.sum(ap== -1))
    akb=[]
    ske=[]
    for i in range(-255,255):
        akb.append(i)
        ske.append(np.sum(ap==i))

    fig=plt.figure()
    plt.suptitle('PEs in red: '+name)
    ax=fig.gca()

    xlim(-20,+20)
    plt.plot(akb,ske,color='green',label='new preditor',alpha=0.5)
    plt.scatter(akb,ske,marker='*')

    ap=pError1[:,:,0]
    ap=np.array(ap)
    # print(np.sum(ap== -1))
    akb=[]
    ske=[]
    for i in range(-255,255):
        akb.append(i)
        ske.append(np.sum(ap==i))

    ax=fig.gca()

    xlim(-20,+20)
    plt.plot(akb,ske,color='red',label='original',alpha=0.5)
    plt.scatter(akb,ske,marker='*')

    plt.legend(loc='upper right')
    plt.savefig(histdir)
    plt.show()

    
    ap=pError[:,:,2]
    ap=np.array(ap)
    # print(np.sum(ap== -1))
    akb=[]
    ske=[]
    for i in range(-255,255):
        akb.append(i)
        ske.append(np.sum(ap==i))

    fig=plt.figure()
    plt.suptitle('PEs in blue: '+name)
    ax=fig.gca()

    xlim(-20,+20)
    plt.plot(akb,ske,color='green',label='new preditor',alpha=0.5)
    plt.scatter(akb,ske,marker='*')

    ap=pError1[:,:,2]
    ap=np.array(ap)
    # print(np.sum(ap== -1))
    akb=[]
    ske=[]
    for i in range(-255,255):
        akb.append(i)
        ske.append(np.sum(ap==i))

    ax=fig.gca()

    xlim(-20,+20)
    plt.plot(akb,ske,color='red',label='original',alpha=0.5)
    plt.scatter(akb,ske,marker='*')

    plt.legend(loc='upper right')
    plt.savefig(histdir_blue)
    plt.show()
    # plt.figure(figsize=(12, 6)), plt.suptitle('error hist')
    # plt.hist(pError[:,:,2].ravel(),alpha = 0.5,bins=40,label='diamond')
    # # plt.figure(figsize=(12, 6)), plt.suptitle('error hist')
    # plt.hist(pError1[:,:,2].ravel(),alpha = 0.5,bins=40,label='original')
    # # plt.hist(pError[:,:,2], alpha = 0.5, label='a')
    # # plt.hist(pError[:,:,2], alpha = 0.5, label='b')
    # plt.legend(loc='upper left')
    # plt.savefig('d-o-hist-cartoon.png')
    img[:,:,0] = predict[:,:,0]
    img[:,:,2] = predict[:,:,2]
    cv2.imwrite(predir, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.show()

if __name__ == '__main__':
    redir='result_images\\'
    srcdir='test_images\\'

    lname='Lena'
    lena=srcdir+'LenaRGB.bmp'
    lena_hist=redir+'Lena_hist.png'
    lena_histb=redir+'Lena_hist_blue.png'
    lena_predict=redir+'Lena_predict.png'

    bname='Baboon'
    baboon=srcdir+'BaboonRGB.bmp'
    baboon_hist=redir+'Baboon_hist.png'
    baboon_histb=redir+'Baboon_hist_blue.png'
    baboon_predict=redir+'Lena_predict.png'

    pname='Peppers'
    peppers=srcdir+'PeppersRGB.bmp'
    peppers_hist=redir+'Peppers_hist.png'
    peppers_histb=redir+'Peppers_hist_blue.png'
    peppers_predict=redir+'Peppers_predict.png'

    cname='Cartoon'
    cartoon=srcdir+'Cartoon.png'
    cartoon_hist=redir+'Cartoon_hist.png'
    cartoon_histb=redir+'Cartoon_hist_blue.png'
    cartoon_predict=redir+'Cartoon_predict.png'


    # cname='test'
    # cartoon='124.png'
    # cartoon_hist=redir+'test_hist.png'
    # cartoon_histb=redir+'test_hist_blue.png'
    # cartoon_predict=redir+'test_predict.png'

    test(512,lname,lena,lena_hist,lena_histb,lena_predict)

    test(512,bname,baboon,baboon_hist,baboon_histb,baboon_predict)
    test(512,pname,peppers,peppers_hist,peppers_histb,peppers_predict)
    test(64,cname,cartoon,cartoon_hist,cartoon_histb,cartoon_predict)

   