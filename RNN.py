__author__ = 'CP'
from network import LSTM
import tensorflow as tf
import argparse
import logging
import time
import datetime
import os
from utils import util

# class RNN:
#     def __init__(self,model)
#     self.model = model
#
#     def forward(self, )
#         return
#
#     def train(self,):

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Pytorch network')
    parser.add_argument('--model_size', type=int, nargs = '+', default = [2,50])
    parser.add_argument('--hidden_size', type=int, default = 10)
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
    model = LSTM.LSTM(LSTM.LSTMcell, input_size, 10, num_layer_lstm, True, True)
