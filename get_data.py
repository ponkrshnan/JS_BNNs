#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to fetch the data and preprocess if necessary

"""
import numpy as np
import mxnet as mx
import config as cfg

def get_data():    
    if (cfg.dataset == 'histo'):
        trn_ready = np.load('data/histo/training_features.npy')
        trn_ready_lab = np.load('data/histo/training_label.npy')
        test_feat = np.load('data/histo/testing_features.npy')
        test_lab = np.load('data/histo/testing_label.npy')
        #frac = [1,0.8,0.6,0.4,0.2,0]
        start_ind = round(len(trn_ready)*0.2*(cfg.run_no-1))
        end_ind = round(len(trn_ready)*0.2*cfg.run_no)
        vld_feat_mx = mx.nd.array(trn_ready[start_ind:end_ind])
        vld_lab_mx = mx.nd.array(trn_ready_lab[start_ind:end_ind])
        train_feat_mx = mx.nd.array(np.delete(trn_ready,range(start_ind,end_ind),axis=0))
        train_lab_mx = mx.nd.array(np.delete(trn_ready_lab,range(start_ind,end_ind),axis=0))   
        test_feat_mx = mx.nd.array(test_feat)
        test_lab_mx = mx.nd.array(test_lab)
        
    elif (cfg.dataset == 'cifar'):
        def unpickle(file):
          import pickle
          with open(file, 'rb') as fo:
              dict = pickle.load(fo, encoding='bytes')
          return dict
        flag = 0 
        for i in range(5):
          if i!=cfg.run_no-1:
             data = unpickle('data/cifar/data_batch_'+str(i+1))
             if flag==0:
               trn_ready = np.reshape(data[b'data'],(10000,3,32,32)) 
               trn_ready_lab = data[b'labels']
               flag=1
             else:
               trn_ready = np.append(trn_ready,np.reshape(data[b'data'],(10000,3,32,32)),axis=0)
               trn_ready_lab = np.append(trn_ready_lab,data[b'labels'],axis=0)
        
        
        
        data = unpickle('data/cifar/data_batch_'+str(cfg.run_no))
        vldn_ready = np.reshape(data[b'data'],(10000,3,32,32)) 
        vldn_ready_lab = data[b'labels']
        
        
        data = unpickle('data/cifar/test_batch')
        tes_ready = np.reshape(data[b'data'],(10000,3,32,32)) 
        tes_ready_lab = data[b'labels']
        
        
        vld_ready = mx.nd.array(vldn_ready)
        vld_lab_mx = mx.nd.array(vldn_ready_lab)
        vld_ready = vld_ready/255
        vld_feat_mx = vld_ready + mx.nd.random.normal(scale = cfg.noise, shape = vld_ready.shape)
        
        tr_ready = mx.nd.array(trn_ready)
        train_lab_mx = mx.nd.array(trn_ready_lab)
        tr_ready = tr_ready/255
        train_feat_mx = tr_ready + mx.nd.random.normal(scale = cfg.noise, shape = trn_ready.shape)
        
        test_ready = mx.nd.array(tes_ready)
        test_lab_mx = mx.nd.array(tes_ready_lab)
        test_ready = test_ready/255
        test_feat_mx = test_ready + mx.nd.random.normal(scale = cfg.noise, shape = test_ready.shape)

    else:
        print("dataset is", cfg.dataset)
        raise ValueError("Datasets can be 'histo' or 'cifar'")
    
    train_size = train_feat_mx.shape[0]   
    train_dataset = mx.gluon.data.dataset.ArrayDataset(train_feat_mx, train_lab_mx)
    val_dataset = mx.gluon.data.dataset.ArrayDataset(vld_feat_mx, vld_lab_mx)    
    test_dataset = mx.gluon.data.dataset.ArrayDataset(test_feat_mx, test_lab_mx)
    train_data_loader = mx.gluon.data.DataLoader(train_dataset, batch_size=cfg.batch_size)
    val_data_loader = mx.gluon.data.DataLoader(val_dataset, batch_size=cfg.batch_size)
    test_data_loader = mx.gluon.data.DataLoader(test_dataset, batch_size=cfg.batch_size)
    return train_data_loader, val_data_loader, test_data_loader, train_size
