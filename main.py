'''
Author: your name
Date: 2021-01-06 16:43:12
LastEditTime: 2021-01-06 20:55:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /digital/main17.py
'''
# ----------------------------------------------------------------------------------------------------------------------
# region import
import nltk, logging
import pandas as pd
from torch.nn import init
from torchtext import data
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from utils import *
from hyper import *
from model.FocalLoss import *
from model.Classifier import *
import warnings
warnings.filterwarnings("ignore")

print(vars(opt))

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=opt.log_datapath+'logging.log', level=logging.DEBUG, format=LOG_FORMAT)

if torch.cuda.is_available() and opt.gpu >= 0:
    device = torch.device('cuda:' + str(opt.gpu))
else:
    device = torch.device('cpu')
print(torch.cuda.get_device_name(), device)

if opt.need_train:
    model_writer = SummaryWriter(opt.log_datapath+'model_model', comment='model')
    train_writer = SummaryWriter(opt.log_datapath+'model_train', comment='train')
    val_writer = SummaryWriter(opt.log_datapath+'model_val', comment='val')
print('import done')
# endregion

# ----------------------------------------------------------------------------------------------------------------------
# region Hyper-parameters
INPUT_SIZE = 300  # embedding_dim
SEQ_LEN = opt.seq_len
MAX_NB_WORDS = opt.max_nb_words
HIDDEN_SIZE = opt.hidden_size  # lstm_dim
NUM_LAYERS = opt.num_layers
NUM_EPOCHS = opt.num_epochs
BATCH_SIZE = opt.batch_size
NUM_CLASSES = 8
# endregion

# ----------------------------------------------------------------------------------------------------------------------
# region Data set
if opt.need_train:
    X_train, y_train = load_data(opt.train_datapath)
    X_train = X_train[:-13]
    y_train = y_train[:-13]
X_test, y_test = load_data(opt.test_datapath)
X_test = X_test[:-13]
y_test = y_test[:-13]

# 划分train val test
if opt.need_train:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=21/171, random_state=1)
    train = pd.DataFrame({'essay': X_train, 'score': y_train})
    val = pd.DataFrame({'essay': X_val, 'score': y_val})
    train.to_csv("./data/train.csv")
    val.to_csv("./data/val.csv")
test = pd.DataFrame({'essay': X_test, 'score': y_test})
test.to_csv("./data/test.csv")

# torch text 定义 Field https://www.jianshu.com/p/71176275fdc5
ESSAY = data.Field(sequential=True, tokenize=nltk.tokenize.word_tokenize, lower=True, fix_length=SEQ_LEN)  # 不够长
SCORE = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
# torch text 定义 Data set
train, val, test = data.TabularDataset.splits(
    path="./data", train='train.csv', validation='val.csv', test='test.csv', format='csv',
    skip_header=True, fields=[('essay_id', None), ('essay', ESSAY), ('score', SCORE)])

# torch text 建立 vocab
ESSAY.build_vocab(train, max_size = 3000, vectors='glove.6B.300d',)
ESSAY.vocab.vectors.unk_init = init.xavier_uniform
# endregion

# ----------------------------------------------------------------------------------------------------------------------
# region Data loader
# torch text 构造 iter
if opt.need_train:
    train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.Phrase), shuffle=True)
    val_iter = data.BucketIterator(val, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.Phrase), shuffle=False)
test_iter = data.BucketIterator(test, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.Phrase), shuffle=False)
# endregion

# ----------------------------------------------------------------------------------------------------------------------
# region Define the model
if opt.need_train:
    model = Classifier(opt).to(device)
    model.embedding_layer.weight.data.copy_(ESSAY.vocab.vectors)
    dummy_input = torch.zeros((opt.batch_size, opt.seq_len), dtype=torch.long).to(device)
    model_writer.add_graph(model, (dummy_input, ))
else:
    model = torch.load(opt.weight_datapath+"model.pt")
    model.state_dict = torch.load(opt.weight_datapath+'./state.pt')
print('define model done')
# endregion

# ----------------------------------------------------------------------------------------------------------------------
# region Criterion & Optimizer
if opt.need_train:
    if opt.loss == 'Focal':
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=2e-6, amsgrad=False)
    print('criterion & Optimizer done')
# endregion

# ----------------------------------------------------------------------------------------------------------------------
# region Train the model
if opt.need_train:
    total_step = len(train_iter)
    history_plot = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        history_train = []
        for i, batch in enumerate(train_iter):
            essay = batch.essay.T.to(device)
            score = torch.as_tensor(batch.score, dtype=torch.long).resize(BATCH_SIZE).to(device)
            # Forward pass
            outputs = model(essay)
            loss = criterion(outputs, score)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Display
            history_train.append(loss.item())
        train_loss = sum(history_train) / len(history_train)

        model.eval()
        history_val = []
        for i, batch in enumerate(val_iter):
            essay = batch.essay.T.to(device)
            score = torch.as_tensor(batch.score, dtype=torch.long).resize(BATCH_SIZE).to(device)
            # Forward pass
            outputs = model(essay)
            loss = criterion(outputs, score)
            # Display
            history_val.append(loss.item())
        val_loss = sum(history_val) / len(history_val)
        train_writer.add_scalar('loss', train_loss, epoch)
        val_writer.add_scalar('loss', val_loss, epoch)
        print(f'Epoch [{epoch + 1:<3}/ {NUM_EPOCHS}], Step [{total_step} / {total_step}], TrainLoss: {train_loss:.4f}, ValLoss: {val_loss:.4f}')

    # torch.save(model, opt.weight_datapath+"model.pt")
    # torch.save(model.state_dict(), opt.weight_datapath+'./state.pt')
# endregion

# ----------------------------------------------------------------------------------------------------------------------
# region Test the model
correct = 0.0
total = 0.0
for i, batch in enumerate(test_iter):
    essay = batch.essay.T.to(device)
    score = torch.as_tensor(batch.score, dtype=torch.long).resize(BATCH_SIZE).to(device)
    outputs = torch.max(model(essay), dim=1).indices
    correct += torch.sum(score==outputs)
    total += BATCH_SIZE
print(f'{correct/total:.2f}')


# tensorboard --logdir=/home/deeplearning/kk/digital/log