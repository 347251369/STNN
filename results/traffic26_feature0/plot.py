import numpy as np
import matplotlib.pyplot as plt

arg = np.load('arg.npy')
t,dy = arg[0],arg[1]
value = np.load('value.npy')
Pearson,RMSE = value[0],value[1]
Y = np.load('Y.npy')

                  
our = np.load('our_Ypred.npy')
AR = np.load('AR_Ypred.npy')
ARIMA = np.load('ARIMA_Ypred.npy')
KAE = np.load('KAE_Ypred.npy')
SVR = np.load('SVR_Ypred.npy')
RNN = np.load('RNN_Ypred.npy')
RBF = np.load('RBF_Ypred.npy')

min_value = min(Y.min(),our.min())
max_value = max(Y.max(),our.max())
#******************************************************************************
# draw pic
#******************************************************************************
#legend
plt.title("Pearson :"+ str(round(Pearson,4)) +"  RMSE :" + str(round(RMSE,4)),weight='bold',fontsize=14)
plt.xlabel('Time(step)',fontsize=12)
plt.ylabel('Value',fontsize=12)
plt.xlim(xmin=0,xmax=t)
plt.ylim(ymin=min_value-3, ymax=max_value+3)
# draw line
plt.plot(np.arange(t-dy+1,t+1,1), np.concatenate([Y[t-dy:t-dy+1],RBF]),color='gray', linestyle='-',marker = "v", MarkerSize=2) #倒三角
plt.plot(np.arange(t-dy+1,t+1,1), np.concatenate([Y[t-dy:t-dy+1],RNN]),color='gray', linestyle='-',marker = "^", MarkerSize=2) #正三角
plt.plot(np.arange(t-dy+1,t+1,1), np.concatenate([Y[t-dy:t-dy+1],SVR]),color='gray', linestyle='-',marker = "s", MarkerSize=2) #正方形
plt.plot(np.arange(t-dy+1,t+1,1), np.concatenate([Y[t-dy:t-dy+1],KAE]),color='gray', linestyle='-',marker = "p", MarkerSize=2)  #五边形
plt.plot(np.arange(t-dy+1,t+1,1), np.concatenate([Y[t-dy:t-dy+1],ARIMA]),color='gray', linestyle='-',marker = "d", MarkerSize=2) #菱形
plt.plot(np.arange(t-dy+1,t+1,1), np.concatenate([Y[t-dy:t-dy+1],AR]),color='gray', linestyle='-',marker = "o", MarkerSize=2) #圆形

plt.plot(np.arange(1,t+1,1), Y, color='blue', linestyle='-',marker = "o", MarkerSize=3)
plt.plot(np.arange(t-dy+1,t+1,1), np.concatenate([Y[t-dy:t-dy+1],our]),color='red', linestyle='-',marker = "o", MarkerSize=3)
plt.ylim(8,75)
plt.axvspan(0, t-dy+1, facecolor='#a2cffe', alpha=0.5)
plt.savefig('1.png', dpi=1000)
plt.show()