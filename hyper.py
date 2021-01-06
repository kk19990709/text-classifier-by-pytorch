'''
Author: your name
Date: 2021-01-06 17:40:07
LastEditTime: 2021-01-06 21:03:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /digital/HyberPara.py
'''
import argparse
parser = argparse.ArgumentParser(description='TM_PATH')
parser.add_argument('--nt', dest='need_train', action='store_true')
parser.add_argument('--bs', dest='batch_size', type=int, default=32)
parser.add_argument('--hs', dest='hidden_size', type=int, default=512)
parser.add_argument('--ne', dest='num_epochs', type=int, default=50)
parser.add_argument('--nl', dest='num_layers', type=int, default=2)
parser.add_argument('--sl', dest='seq_len', type=int, default=200)
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.004)
parser.add_argument('--dr', dest='dropout_rate', type=float, default=0.05)
parser.add_argument('--mnw', dest='max_nb_words', type=int, default=3000)
parser.add_argument('--gpu', dest='gpu', type=int, default=2)
parser.add_argument('--loss', dest='loss', type=str, default='Focal')
parser.add_argument('--trdp', dest='train_datapath', type=str, default='./data/r8-train-all-terms.txt')
parser.add_argument('--tedp', dest='test_datapath', type=str, default='./data/r8-test-all-terms.txt')
parser.add_argument('--csvdp', dest='csv_datapath', type=str, default='./data/')
parser.add_argument('--logdp', dest='log_datapath', type=str, default='./log/')
parser.add_argument('--wedp', dest='weight_datapath', type=str, default='./weight/')
opt = parser.parse_args()

INPUT_SIZE = 300
NUM_CLASSES = 8