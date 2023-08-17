__author__ = 'CP'
from network import LSTM
import argparse
import logging
import time
import datetime
import os
from utils import util
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
# class RNN:
#     def __init__(self,model)
#     super(self,)
#     self.model = model
#     self.fc = torch.nn.Linear(self.model)
#
#     def forward(self, )
#         #reserved as the attention module interface
#         return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'torch implement')
    parser.add_argument('--model_size', type=int, nargs = '+', default = [2,50])
    parser.add_argument('--hidden_size', type=int, default = 10)
    parser.add_argument('--num_layer', type=int, default = 2)
    parser.add_argument('--epochs', type=int, default = 10)
    parser.add_argument('--train_dir', type = str, default = './logs/')
    FLAGS, unparsed = parser.parse_known_args()

    train_dir = FLAGS.train_dir + time.strftime('%Y%m%d_%H%M%S') + '_temp'
    if not os.path.exists(train_dir):
        # os.makedirs(train_dir)
        os.mkdir(train_dir)

    logging.basicConfig(level = logging.INFO, filename = train_dir + "/train_{}.log".format(time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))), filemode = "w")
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>  Running at %s', datetime.datetime.now())
    util.show_param(FLAGS, logger, stream_handler)

    input_size = 3
    batch_size = 32
    # time_step = 2
    time_step = 4
    output_size = 1

    num_layer = FLAGS.num_layer
    hidden_size = FLAGS.hidden_size
    epoch = FLAGS.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = LSTM.LSTM(LSTM.LSTMcell, num_layer, input_size, hidden_size).to(device)
    print("line init....i")
    for param in net.state_dict():
        print("net.parameters()  ",param, net.state_dict()[param].shape)
        # print("net.parameters()  ",param)
    # net=RegLSTM(input_size,output_size,mid_dim,mid_layers,True).to(device)
    criterion=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-2)

    data = util.load_data()
    print("data.shape:  ",data.shape)
    train_size = int(len(data) * 0.75)
    data_sample = np.zeros((train_size - time_step + 1, time_step, input_size))
    label_sample = np.zeros((train_size - time_step + 1, time_step, output_size))
    for i in range(train_size - time_step + 1):
        data_sample[i] = data[i:i + time_step, :]
        label_sample[i] = data[i + 1:i + 1 + time_step, 0:1:]

    for i in range(epoch):
        for j in range(int((train_size - time_step + 1) / batch_size)):

            train_X = data_sample[j * batch_size:(j + 1) * batch_size, :, :]
            train_Y = label_sample[j * batch_size:(j + 1) * batch_size, :, :]
            var_x = torch.tensor(train_X, dtype=torch.float32, device=device)
            var_y = torch.tensor(train_Y, dtype=torch.float32, device=device)
            print("input.shape: ", var_x.shape)
            out = net(var_x)
            print("Step j: ", j)
            loss = criterion(out, var_y)
            # loss = criterion(out[:,-1,:], var_y[:,-1,:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_X = data_sample[(j + 1) * batch_size:, :, :]
        train_Y = label_sample[(j + 1) * batch_size:, :, :]
        var_x = torch.tensor(train_X, dtype=torch.float32, device=device)
        var_y = torch.tensor(train_Y, dtype=torch.float32, device=device)
        out = net(var_x)
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 2 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(i, loss.item()))


    net=net.eval()
    test_X=data[train_size:,:]
    test_Y=data[train_size+time_step:,0:1:]
    test_y=list()

    for i in range(test_X.shape[0]-time_step):
        test_x=test_X[i:time_step+i,:].reshape(1,time_step,input_size)
        test_x=torch.tensor(test_x,dtype=torch.float32,device=device)
        tem=net(test_x).cpu().data.numpy()
        test_y.append(tem[0][-1])

    test_y=np.array(test_y).reshape((-1,1))
    diff=test_y-test_Y
    l1_loss=np.mean(np.abs(diff))
    l2_loss=np.mean(diff**2)
    print("L1:{:.3f}    L2:{:.3f}".format(l1_loss,l2_loss))
    plt.plot(test_y, 'r', label='pred')
    plt.plot(test_Y, 'b', label='real', alpha=0.3)


    a=np.array([1,2,3])
    b=np.array([2,2,4])

    a=a.reshape(-1,1)
    b=b.reshape(-1,1)

    np.corrcoef(a,b)
    print("Line 187")
    a.reshape((-1))
    print("..........")
