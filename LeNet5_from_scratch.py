import pickle
import json  # save loss, acc
import random
import numpy as np
import matplotlib.pyplot as plt
# from abc import ABCMeta, abstractmethod

#################################################
import cv2
import math

import os
import multiprocessing as mp
from tqdm import tqdm
# import time

from LeNet5 import *
from memory_profiler import profile
cpus = mp.cpu_count()
print(cpus)
NUM_PROCESSES = 8

# load data
resized_train_imgs = np.load(file="./data/resized_train_imgs.npy")
resized_val_imgs = np.load(file="./data/resized_val_imgs.npy")
resized_test_imgs = np.load(file="./data/resized_test_imgs.npy")
train_label = np.load(file="./data/train_label.npy")
val_label = np.load(file="./data/val_label.npy")
test_label = np.load(file="./data/test_label.npy")

def MoveColorChannel(image: np.ndarray) -> np.ndarray:
    return np.moveaxis(image, source=2, destination=0)  # reshape (H, W, C) to (C, H, W)

if __name__ == '__main__':
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        resized_train_imgs = pool.map(MoveColorChannel, tqdm(resized_train_imgs))
        resized_val_imgs = pool.map(MoveColorChannel, tqdm(resized_val_imgs))
        resized_test_imgs = pool.map(MoveColorChannel, tqdm(resized_test_imgs))

# convert list to numpy.ndarray
resized_train_imgs = np.array(resized_train_imgs)
resized_val_imgs = np.array(resized_val_imgs)
resized_test_imgs = np.array(resized_test_imgs)
# list to numpy.ndarray
train_label = np.array(train_label)
val_label = np.array(val_label)
test_label = np.array(test_label)


"""
(1) Prepare Data: Load, Shuffle, Normalization, Batching, Preprocessing
"""

X_train, Y_train, X_val, Y_val, X_test, Y_test = resized_train_imgs, train_label, resized_val_imgs, val_label, resized_test_imgs, test_label
# X_train, Y_train, X_val, Y_val, X_test, Y_test = resized_train_imgs[::25,:,:,:], train_label[::25], resized_val_imgs[::25,:,:,:], val_label[::25], resized_test_imgs[::25,:,:,:], test_label[::25]
X_train, X_val, X_test = X_train/float(255), X_val/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_val -= np.mean(X_val)
X_test -= np.mean(X_test)

batch_size = 64
# D_in = 784
D_out = 50

# print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))
print("batch_size: " + str(batch_size) + ", D_out: " + str(D_out))

def acc_fn(Y_pred, Y_true):
    return np.mean(np.argmax(Y_pred, axis=1)==np.argmax(Y_true, axis=1))

### Lenet Forward Test ###
train_dataloader = DataLoader(data=X_train,
                              labels=Y_train,
                              batch_size=64,
                              shuffle=True)
val_dataloader = DataLoader(data=X_val,
                            labels=Y_val,
                            batch_size=64,
                            shuffle=True)

model = LeNet5()
# optim = SGDMomentum(params=model.get_params(), lr=1e-3, momentum=0.99, reg=0)
optim = SGD(params=model.get_params(), lr=1e-3, reg=0)
criterion = CrossEntropyLoss()

# Train
# https://zhuanlan.zhihu.com/p/121003986
# https://github.com/pythonprofilers/memory_profiler
@profile(precision=4, stream=open('./memory/memory_profiler_scratch_model.log','w+'))
def train(model, train_dataloader, n_epochs):
    # EPOCHS = 5
    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []

    for i in range(n_epochs):
        print(f"epoch: {i+1}")
        # train
        train_loss, train_acc = 0, 0
        with tqdm(total=train_dataloader.num_batches) as pbar:
            for X_batch, Y_batch in train_dataloader:
                # get batch, make onehot
                # X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
                Y_batch = MakeOneHot(Y_batch, D_out)

                # forward, loss, backward, step
                Y_pred = model.forward(X_batch)
                loss, _ = criterion.get(Y_pred, Y_batch)  # loss, dout
                dout = Y_pred - Y_batch  # pred - label
                model.backward(dout)
                optim.step()

                # train accuracy
                acc = acc_fn(Y_pred, Y_batch)

                train_loss += loss
                train_acc += acc
                pbar.update(1)

        train_loss /= train_dataloader.num_batches
        train_acc /= train_dataloader.num_batches
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
                
        if i % 1 == 0:
            print("%s%% epoch: %s, train loss: %s" % (round(100*(i+1)/n_epochs, 4), i+1, round(train_loss, 4)))


        # validation
        val_loss, val_acc = 0, 0
        for X_batch, Y_batch in val_dataloader:
            # get batch, make onehot
            # X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
            Y_batch = MakeOneHot(Y_batch, D_out)

            # forward, loss, backward, step
            Y_pred = model.forward(X_batch)
            loss, _ = criterion.get(Y_pred, Y_batch)  # loss, dout
            dout = Y_pred - Y_batch  # pred - label
            model.backward(dout)
            optim.step()

            # train accuracy
            acc = acc_fn(Y_pred, Y_batch)

            val_loss += loss
            val_acc += acc

        val_loss /= val_dataloader.num_batches
        val_acc /= val_dataloader.num_batches
        val_losses.append(val_loss)
        val_accuracy.append(val_acc)

    return train_losses, val_losses, train_accuracy, val_accuracy

train_losses, val_losses, train_accuracy, val_accuracy = train(model, train_dataloader, n_epochs=5)

# inference time
import timeit
net = LeNet5()
elapsed_time = timeit.timeit(lambda: net.forward(X_test), number=10)
print(f"Inference time: {elapsed_time:.8f} seconds")

# space complexity
def count_parameters(model):
    # 計算參數數量
    num_params = 0
    for param in model.get_params():
        num_params += np.prod(param['val'].shape)
    return num_params

input_shape = (64, 3, 32, 32)
net = LeNet5()

# 計算參數數量
num_params = count_parameters(model)
print("Number of parameters:", num_params)

