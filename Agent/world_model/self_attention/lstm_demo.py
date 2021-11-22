import torch
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F
import torch.optim as optim

decay_lr_factor = 0.3
decay_lr_every = 5
lr = 0.01

rnn = nn.LSTM(4, 8, 3) 

input = torch.tensor([[[1,1,1,1], [2,2,2,2],[-1,-1,-1,-1]], [[2,2,2,2], [1,2,3,1], [-2,-1,-1,-1]]], dtype=torch.float)
# input = torch.randn(5, 2, 4)
# print("input",input)
h0 = torch.randn(3, 3, 8)
c0 = torch.randn(3, 3, 8)

optimizer = optim.Adam(rnn.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

rnn.train()
for j in range(0,100):
    y = torch.tensor([[1,2,3,4], [-1,-4,-3,-4]], dtype=torch.float)         
    optimizer.zero_grad()
    yn, (output, cn) = rnn(input, (h0, c0))
    print("output",output)

    loss = F.mse_loss(output, y)
    print("Test_Loss",loss)
    loss.backward()
    # acc_loss += batch_size * loss.item()
    # num_samples += y.shape[0]
    optimizer.step()


