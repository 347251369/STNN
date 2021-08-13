import torch
from torch import nn
import numpy as np

def train(model, train_loader, lr, weight_decay, num_epochs, epoch_update, steps=1, steps_back=1, gradclip=1):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
             
            
    def lr_scheduler(optimizer, epoch, decayEpoch=[]):
                    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.2
                        return optimizer
                    else:
                        return optimizer


    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        for batch_idx, data_list in enumerate(train_loader):
            model.train()
            out, out_back = model(data_list[0], mode='forward')

            for k in range(steps):
                if k == 0:
                    loss_fwd = criterion(out[k], data_list[k+1])
                else:
                    loss_fwd += criterion(out[k], data_list[k+1])

            
            loss_identity = criterion(out[-1], data_list[0]) * steps

            loss_bwd = 0.0
            loss_consist = 0.0

            loss_bwd = 0.0
            loss_consist = 0.0

            out, out_back = model(data_list[-1], mode='backward')

            for k in range(steps_back):
                
                if k == 0:
                    loss_bwd = criterion(out_back[k], data_list[::-1][k+1])
                else:
                    loss_bwd += criterion(out_back[k], data_list[::-1][k+1])

  
                A = model.dynamics.dynamics.weight
                B = model.backdynamics.dynamics.weight

                K = A.shape[-1]

                for k in range(1,K+1):
                    As1 = A[:,:k]
                    Bs1 = B[:k,:]
                    As2 = A[:k,:]
                    Bs2 = B[:,:k]

                    Ik = torch.eye(k).float()

                    if k == 1:
                        loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                         torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                    else:
                        loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                         torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k)

            loss = loss_fwd + loss_identity +  0.1 * loss_bwd + 0.2 * loss_consist

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
            optimizer.step()           

        # schedule learning rate decay    
        lr_scheduler(optimizer, epoch, decayEpoch=epoch_update)
        
        if (epoch) % 20 == 0:
                print('********** Epoche %s **********' %(epoch+1))
                print("loss identity: ", loss_identity.item())
                print("loss backward: ", loss_bwd.item())
                print("loss consistent: ", loss_consist.item())
                print("loss forward: ", loss_fwd.item())
                print("loss sum: ", loss.item())


    return model
