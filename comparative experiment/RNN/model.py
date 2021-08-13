from torch import nn
import torch

class RNN(nn.Module):

    def __init__(self, input_size,hidden_size,output_size):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,                         #feature_len=1
            hidden_size=hidden_size,                       #隐藏记忆单元尺寸hidden_len
            num_layers=1,                                  #层数
            batch_first=True,                              #在传入数据时,按照[batch,seq_len,feature_len]的格式
        )
        for p in self.rnn.parameters():                    #对RNN层的参数做初始化
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)  #输出层


    def forward(self, x, hidden_prev):
        '''
        x：一次性输入所有样本所有时刻的值(batch,seq_len,feature_len)
        hidden_prev：第一个时刻空间上所有层的记忆单元(batch,num_layer,hidden_len)
        输出out(batch,seq_len,hidden_len)和hidden_prev(batch,num_layer,hidden_len)
        '''
        out, hidden_prev = self.rnn(x, hidden_prev)
        s,b,h=out.shape
        out = out.view(s*b, h)    #[batch=1,seq_len,hidden_len]->[seq_len,hidden_len]
        out = self.linear(out)             #[seq_len,hidden_len]->[seq_len,feature_len=1]
        out = out.unsqueeze(dim=0)         #[seq_len,feature_len=1]->[batch=1,seq_len,feature_len=1]
        return out, hidden_prev