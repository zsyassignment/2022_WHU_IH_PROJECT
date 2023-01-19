import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from pokemon import Pokemon
# from resnet import ResNet18
# from model.predict_model import PredictModel
from model.cnnp import CNNP
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter   

batchsz = 4
lr = 1e-3
epochs = 16
device = torch.device('cuda')
torch.manual_seed(1234)
train_db = Pokemon('traincases\\', 2000, mode='train')
val_db = Pokemon('traincases\\', 2000, mode='val')
test_db = Pokemon('traincases\\', 2000, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=0)
val_loader = DataLoader(val_db, batch_size=1, num_workers=0)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)
model = CNNP().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.005)
criterion = torch.nn.MSELoss(reduction='mean')
def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    a=0
    for i,(x, y) in enumerate(val_loader):
        x=x.to(torch.float32)
        y=y.to(torch.float32)
        x = x.squeeze(dim=0)  # 成员函数删除第二维度
        x, y = x.to(device), y.to(device)
        loss=0
        cnt=0
        with torch.no_grad():
            cnt+=1
            logits = model(x)
            # pred = logits.argmax(dim=1)
            pred = criterion(y,torch.flatten(logits))
            # print(type(pred.item()))
            loss+=pred.item()

    return pred/cnt

def l2_penalty(w):
    return (np.array(w)**2).sum() / 2

def main():
    
    log_writer = SummaryWriter('log\\')
    # reg_criterion = l2_penalty()
    # strength = 0.01
    print("loading data...")
    best_acc, best_epoch = 1000000, 0
    global_step = 0
    round = 0
    trainl=[]
    testl=[]
    akb=[]
    # for i, data in enumerate(datas):
	# # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
    #     print("第 {} 个Batch \n{}".format(i, data))
    print("begin training")
    epochcnt=0
    for epoch in range(epochs):
        epochcnt+=1
        for step, (a,b) in enumerate(train_loader):
            a=a.to(torch.float32)
            b=b.to(torch.float32)

            round+=1
            # print('batch '+str(round))
            x1 = a[0,:,:,:,:]
            y1 = b[0,:]
            x2 = a[1,:,:,:,:]
            y2 = b[1,:]
            x3 = a[2,:,:,:,:]
            y3 = b[2,:]
            x4 = a[3,:,:,:,:]
            y4 = b[3,:]
            x1, y1 = x1.to(device), y1.to(device)
            x2, y2 = x2.to(device), y2.to(device)
            x3, y3 = x3.to(device), y3.to(device)
            x4, y4 = x4.to(device), y4.to(device)
            logits1 = model(x1)
            logits2 = model(x2)
            logits3 = model(x3)
            logits4 = model(x4)
            logits1=torch.flatten(logits1)
            logits2=torch.flatten(logits2)
            logits3=torch.flatten(logits3)
            logits4=torch.flatten(logits4)
            loss1 = criterion(logits1, y1)
            loss2 = criterion(logits2, y2)
            loss3 = criterion(logits3, y3)
            loss4 = criterion(logits4, y4)
            loss = (loss1+loss2+loss3+loss4)/4
            val_acc = evaluate(model, val_loader)
            # print('train loss:'+str(loss.item())+'      test loss '+str(val_acc.item()))
            log_writer.add_scalar('loss/Train', loss.item(), round) 
            log_writer.add_scalar('loss/Test', val_acc.item(), round) 
            akb.append(round)
            trainl.append(loss)
            testl.append(val_acc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            global_step += 1
            if val_acc < best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best.mdl')
        print('epoch ',epochcnt)
        print('train loss:'+str(loss.item())+'      test loss '+str(val_acc.item()))
            # for k in range((a.size())[0]):
                
            #     # print('x',x.size())
            #     # print('y',y.size())
            #     x = a[k,:,:,:,:]
            #     y = b[k,:]
            #     x, y = x.to(device), y.to(device)
            #     # print('x',x.size())
            #     # print('y',y.size())
            #     logits = model(x)
            #     # print(logits.size())
            #     logits=torch.flatten(logits)
            #     # print(logits.size())
            #     loss = criterion(logits, y)
            #     # reg_loss = l2_penalty(model.weight())
            #     # loss += strength * reg_loss
            #     val_acc = evaluate(model, val_loader)
            #     print('train loss:'+str(loss.item())+'      test loss '+str(val_acc))
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step() 
            #     global_step += 1
            # del(a)
            # del(b)

        # if epoch % 2 == 0:
        # val_acc = evaluate(model, val_loader)
    fig=plt.figure()
    plt.suptitle('loss in train set and test set')
    ax=fig.gca()
    plt.plot(akb,trainl,color='green',label='train loss',alpha=0.5)
    plt.plot(akb,testl,color='red',label='test loss',alpha=0.5)
    plt.legend(loc='upper right')
    plt.savefig('losses.png')
    plt.show()
    print('best acc:',best_acc,'best epoch:', best_epoch)
    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt')
    test_acc = evaluate(model, test_loader)
    print('test acc:', test_acc)
if __name__ == '__main__':
    main()
