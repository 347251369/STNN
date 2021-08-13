import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from read_dataset import *
from model import *
from train import *
import os
#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--model', type=str, default='RNN', metavar='N', help='model')
parser.add_argument('--seed', type=int, default='1',  help='seed value')
parser.add_argument('--lr', type=float, default=1e-2, metavar='N', help='learning rate (default: 0.01)')
parser.add_argument('--lr_update', type=int, nargs='+', default=[50, 100, 150, 200], help='decrease learning rate at these epochs')
parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--wd', type=float, default=1e-5, metavar='N', help='weight_decay (default: 1e-5)')
parser.add_argument('--gradclip', type=float, default=1e-5, help='gradient clipping')
parser.add_argument('--hidden_size', type=int, default=10, help='Hidden layer size')
#
parser.add_argument('--batch', type=int, default=16, metavar='N', help='batch size (default: 8)')
#
parser.add_argument('--dataset', type=str, default='lorenz', metavar='N', help='dataset')
#
parser.add_argument('--dim', type=int, default='0',  help='prediction dimension')
#
parser.add_argument('--dy', type=int, default=20, metavar='N', help='dimension of y')
#
parser.add_argument('--feature', type=int, default=0, help='Is feature selected')
#
parser.add_argument('--noise', type=float, default=0.0, help='noise of data')
#
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

#******************************************************************************
# load data
#******************************************************************************
X = data_from_name(args.dataset,args.dim,args.feature,args.noise)
t,dx =X.shape[0],X.shape[1]
###### scale #####
Xmax, Xmin = np.max(X), np.min(X)
X = ((X-Xmin)/(Xmax-Xmin)+0.01)*0.9
##### split into train and test set #####
Xtrain = X[0:t-args.dy+1]
Xtest = X[t-args.dy+1:]

Y = X[:,0]
Ytest = Y[t-args.dy+1:]
#******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
#******************************************************************************
Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, Xtrain.shape[1],1)
Xtest = Xtest.reshape(Xtest.shape[0], 1, Xtest.shape[1],1)
Xtrain,Xtest = torch.from_numpy(Xtrain).float().contiguous(),torch.from_numpy(Xtest).float().contiguous()
#******************************************************************************
# Create Dataloader objects
#******************************************************************************
train_data = torch.utils.data.TensorDataset(Xtrain[:-1,:,:,:], Xtrain[1:,:,:,:])
train_loader = DataLoader(dataset = train_data, batch_size = args.batch)# shuffle = True
#==============================================================================
# Model
#==============================================================================
model = RNN(dx,args.hidden_size,dx)
print('**** Setup ****')
print('Total params: %.2fk' % (sum(p.numel() for p in model.parameters())/1000.0))
print('************')
print(model)
#==============================================================================
# Start training
#==============================================================================
model,hidden_prev = train(model,train_loader,lr=args.lr,weight_decay=args.wd,num_epochs=args.epochs,
	epoch_update=args.lr_update,gradclip=args.gradclip,hidden_size=args.hidden_size)
#******************************************************************************
# Prediction
#******************************************************************************
model.eval()
Ypred = []
Ytmp = Xtrain[Xtrain.shape[0]-1:Xtrain.shape[0],:,:,:]
Ytmp = Ytmp.float().view(1, Ytmp.shape[0], Ytmp.shape[2])
for i in np.arange(0,args.dy-1,1):
    Ytmp, hidden_prev = model(Ytmp, hidden_prev)
    Ypred.append(Ytmp.detach().numpy().ravel()[0])
Ypred=np.array(Ypred)
#==============================================================================
# result
#==============================================================================
Y = (Y/0.9 - 0.01) * (Xmax - Xmin) + Xmin
Ypred = (Ypred/0.9 - 0.01) * (Xmax - Xmin) + Xmin
Ytest = (Ytest/0.9 - 0.01) * (Xmax - Xmin) + Xmin

Pearson = np.corrcoef(Ypred, Ytest)[0,1]
RMSE = np.sqrt(sum((Ypred-Ytest)**2)/len(Ypred))
StdY = Y[t-2*args.dy+2:t]
RMSE = RMSE / np.sqrt(sum((StdY-sum(StdY)/len(StdY))**2)/len(StdY))
#******************************************************************************
# Save data
#******************************************************************************
address = '../../results/' + args.dataset + str(args.dim) + '_feature' + str(args.feature) + (('_noise' + str(args.noise)) if args.dataset == "lorenz" else '')
if not os.path.exists(address):
	os.makedirs(address)

np.save(address +'/'+args.model+'_Ypred.npy', Ypred)
#******************************************************************************
# draw pic
#******************************************************************************
#legend
plt.title("Pearson :"+ str(round(Pearson,4)) +"  RMSE :" + str(round(RMSE,4)))
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim(xmin=0,xmax=t)
plt.ylim(ymin=min(Y.min(),Ypred.min())-0.3, ymax=max(Y.max(),Ypred.max())+0.3)
# draw line
plt.plot(np.arange(1,t-args.dy+2,1), Y[:t-args.dy+1], color='blue', linestyle='-',marker = "o", MarkerSize=3)
plt.plot(np.arange(t-args.dy+1,t+1,1),Y[t-args.dy:t], label='True',color='green', linestyle='-',marker = "o", MarkerSize=3)
plt.plot(np.arange(t-args.dy+1,t+1,1), np.concatenate([Y[t-args.dy:t-args.dy+1],Ypred]),label='Prediction',color='red', linestyle='-',marker = "o", MarkerSize=3)
name = address +'/'+args.model+'.png'
plt.savefig(name, dpi=100)
plt.show()