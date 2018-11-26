#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:27:24 2018

@author: icedeath
"""

#coding=utf

from keras.utils import multi_gpu_model
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import Lambda
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import callbacks
from keras.layers.normalization import BatchNormalization as BN
import argparse
#import scipy.io as sio
import h5py
from keras.layers.advanced_activations import ELU

K.set_image_data_format('channels_last')

def build_CNN(input_shape):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=64, kernel_size=(1,3), strides=(1,1), padding='same')(x)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=64, kernel_size=(1,3), strides=(1,1), padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    
    conv1 = layers.Conv2D(filters=128, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=128, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)

    conv1 = layers.Conv2D(filters=256, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=256, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=256, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    
    conv1 = layers.Conv2D(filters=3, kernel_size=(1,1), strides=1, padding='same')(conv1)
    conv1 = layers.pooling.GlobalAveragePooling2D()(conv1)
    output = layers.Dense(3, activation = 'tanh')(conv1)

    model = models.Model(x, output)
    return model

def build_SAE(input_shape):
    x = layers.Input(shape=input_shape)
    sae1 = layers.Dense(500,activation = 'tanh')(x)
    sae1 = layers.Dense(1000,activation = 'tanh')(sae1)
    sae1 = layers.Dense(1000,activation = 'tanh')(sae1)
    sae1 = layers.Dense(500,activation = 'tanh')(sae1)
    sae1 = layers.Dense(200,activation = 'tanh')(sae1)
    output = layers.Dense(3,activation = 'tanh')(sae1)

    model = models.Model(x, output)
    return model




def margin_loss(y_true, y_pred, margin = 0.9):
    positive_cost = (y_true + 1)/2 * K.cast(
                    K.less(y_pred, margin), 'float32') * K.pow((y_pred - margin), 2)
    negative_cost = (1 - (y_true + 1)/2) * K.cast(
                    K.greater(y_pred, -margin), 'float32') * K.pow((y_pred + margin), 2)
    return 0.5 * positive_cost + 0.5 * negative_cost


def train(model, data, args):
    (x_train, y_train) = data

    checkpoint = callbacks.ModelCheckpoint(args.save_file, monitor='val_loss', verbose=1, save_best_only=True, 
                                  save_weights_only=True, mode='auto', period=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    #model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss= margin_loss,
                  metrics={})
    if args.load == 1:
        model.load_weights(args.save_file)
        print('Loading %s' %args.save_file)
    hist = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
                     validation_split = 0.1, callbacks=[checkpoint, lr_decay])
    return hist.history


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=20, type=int,
                        help="迭代次数")
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--lr', default=0.002, type=float,
                        help="学习率")
    parser.add_argument('--lr_decay', default=0.95, type=float,
                        help="衰减")
    parser.add_argument('-sf', '--save_file', default='dl_demod.h5',
                        help="保存的权重文件")
    parser.add_argument('-t', '--test', default=0,type=int,
                        help="测试模式")
    parser.add_argument('-l', '--load', default=1,type=int,
                        help="如果需要载入模型，设为1")
    parser.add_argument('-p', '--plot', default=0,type=int,
                        help="设为1时，在训练结束后画出loss变化曲线")
    parser.add_argument('-d', '--dataset', default='data_demod.mat',
                        help="数据文件")
    parser.add_argument('-m', '--model', default= 1,type = int,
                        help="1为卷积，2为全连接")
    args = parser.parse_args()
    print(args)
    
    K.set_image_data_format('channels_last')
    
    with h5py.File(args.dataset, 'r') as data:
        for i in data:
            locals()[i] = data[i].value
    y_train = finalimput2.reshape(int(finalimput2.shape[1]/3), 3)
    if args.model == 1:
        x_train = finaldata2.reshape(int(finaldata2.shape[1]/48),1,48,1)
        model = build_CNN(input_shape = x_train.shape[1:])
        print('Training using CNN...')
    if args.model == 2:
        x_train = finaldata2.reshape(int(finaldata2.shape[1]/48),48)
        model = build_SAE(input_shape = x_train.shape[1:])    
        print('Training using SAE...')
        
    model.summary()

    if args.test == 0:    
        history = train(model=model, data=((x_train, y_train)), args=args)
        if args.plot == 1:    
            train_loss = np.array(history['loss'])
            val_loss = np.array(history['val_loss'])
            plt.plot(np.arange(0, args.epochs, 1),train_loss,label="train_loss",color="red",linewidth=1.5)
            plt.plot(np.arange(0, args.epochs, 1),val_loss,label="val_loss",color="blue",linewidth=1.5)
            plt.legend()
            plt.show()
            plt.savefig('loss.png')
    else:
        model.load_weights(args.save_file)
        print('Loading %s' %args.save_file)
      
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Predicting final symbols...')
    y_pred1 = model.predict(x_train, batch_size=args.batch_size,verbose=1)
    y_pred = np.sign(np.reshape(y_pred1, np.prod(y_pred1.shape)))
    y = np.squeeze(finalimput2)
    print('Train acc:', np.sum(y_pred == y)/np.float(y.shape[0]))
    print('-' * 30 + 'End: test' + '-' * 30)   
