
创建虚拟环境

```
conda create -n example python=3.7
pip install -r requirement.txt
```

下载代码

```
<!-- git clone -->
```

进入目录

```
cd /home/example/
```

测试已训练好的模型（准确率96%）

```
python main.py
```

训练新模型（GeForce GTX 1080Ti 11G 每代用时不到30s，15代即可）

```
python main.py --nt
```

查看argparse参数

```
python main.py --h
```

```
usage: main.py [-h] [--nt] [--bs BATCH_SIZE] [--hs HIDDEN_SIZE]
            [--ne NUM_EPOCHS] [--nl NUM_LAYERS] [--sl SEQ_LEN]
            [--lr LEARNING_RATE] [--dr DROPOUT_RATE] [--mnw MAX_NB_WORDS]
            [--gpu GPU] [--loss LOSS] [--trdp TRAIN_DATAPATH]
            [--tedp TEST_DATAPATH] [--csvdp CSV_DATAPATH]
            [--logdp LOG_DATAPATH] [--wedp WEIGHT_DATAPATH]
```

可视化

```
tensorboard --logdir=example/log
```


