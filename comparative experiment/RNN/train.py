import torch
import numpy as np


def train(model, train_loader, lr, epoch_update, weight_decay, num_epochs,hidden_size,gradclip=1):

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    def lr_scheduler(optimizer, epoch,decayEpoch=[]):
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        return optimizer
                    else:
                        return optimizer

    criterion = torch.nn.MSELoss()
    hidden_prev = torch.zeros(1, 1, hidden_size)     #initial memory cell h0
    for epoch in range(num_epochs):
        for idx, (X , Y) in enumerate(train_loader):
            X = X.float().view(1, X.shape[0], X.shape[2])
            Y = Y.float().view(1, Y.shape[0], Y.shape[2])

            model.train()
            loss = torch.tensor(0.0)
            #loss
            out,hidden_prev=model(X,hidden_prev)
            hidden_prev = hidden_prev.detach()
            loss = criterion(out,Y)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip)
            optimizer.step()
        # schedule learning rate decay    
        lr_scheduler(optimizer, epoch, decayEpoch=epoch_update)                
        if (epoch) % 100 == 0:
            print('********** Epoche %s **********' %(epoch+1))
            print("loss : ", loss.item())
            
    return model,hidden_prev