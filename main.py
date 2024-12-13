#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main to execute BNNs

"""

from __future__ import print_function

import mxnet as mx
import numpy as np
import os

from mxnet import gluon, autograd, nd
from tqdm import tqdm

from get_data import get_data
import config as cfg
from resnet_bnns import resnet50_v1, resnet152_v1, resnet18_v1
import bnn_metrics
import bnn_utils

mx.random.seed(1)
ctx = mx.gpu(0) if mx.context.num_gpus()>0 else mx.cpu()

print("Loading dataset...")
#Data set
######################################################################################
train_data_loader, val_data_loader, test_data_loader, train_size = get_data()
#print(train_size)
########################################################################################

# Model
################################################################################
def get_model(model):
    if model == "resnet50":
        net = resnet50_v1(pretrained=True, ctx=ctx)
    elif model == "resnet152":        
        net = resnet152_v1(pretrained=True, ctx=ctx)
    elif model == "resnet18":        
        net = resnet18_v1(pretrained=True, ctx=ctx)
    return net        
    
net = get_model(cfg.model)

with net.name_scope():
  net.output = gluon.nn.Dense(cfg.num_outputs)    

net.output.collect_params().initialize(mx.init.Normal(sigma=10), ctx=ctx)
net.hybridize()

for data,label in train_data_loader:
    data = bnn_utils.resize_data(data)
    data = data.as_in_context(ctx)
    net(data)
    break
###########################

# initialize variational parameters; mean and variance for each weight
temp_mus = list(map(lambda x: x.data(ctx), net.collect_params().values()))
mus = []
rhos = []

shapes = list(map(lambda x: x.shape, net.collect_params().values()))
count = 0
for shape in shapes:
    mu = gluon.Parameter('mu', shape=shape, init=mx.init.Constant(temp_mus[count]))
    rho = gluon.Parameter('rho',shape=shape, init=mx.init.Constant(cfg.rho_offset))
    mu.initialize(ctx=ctx)
    rho.initialize(ctx=ctx)
    mus.append(mu)
    rhos.append(rho)
    count += 1

variational_params = mus + rhos

raw_mus = list(map(lambda x: x.data(ctx), mus))
raw_rhos = list(map(lambda x: x.data(ctx), rhos))

###############################

###############################
if cfg.dataset=='histo':
    step_itr = [len(train_data_loader)*4, len(train_data_loader)*8, len(train_data_loader)*12, len(train_data_loader)*20]
    schedule = mx.lr_scheduler.MultiFactorScheduler(step=step_itr, factor=0.1)
    adam_opt = mx.optimizer.Adam(learning_rate=cfg.learning_rate,lr_scheduler=schedule)
else:
    adam_opt = mx.optimizer.Adam(learning_rate=cfg.learning_rate)
    
trainer = gluon.Trainer(variational_params,optimizer=adam_opt)

if cfg.loss=="JSG":
    bbb_loss = bnn_metrics.BBBLoss_JSG() 
elif cfg.loss=="JSA":
    bbb_loss = bnn_metrics.BBBLoss_JSA()
elif cfg.loss == "KL":
    bbb_loss = bnn_metrics.BBBLoss_KL()
else:
    print("Loss is", cfg.loss)
    raise ValueError("Loss can be 'JSG', 'JSA' or 'KL'")
    

################################

#########
print('Main LOOp')
#########
#prior_mus = raw_mus
#prior_rhos = transform_rhos(raw_rhos)
loss_list = []
mov_loss = []
train_acc_list = []
vld_acc_list = []
max_acc = 0
smoothing_constant = .01

if not os.path.exists(cfg.out_dir):
    os.makedirs(cfg.out_dir, exist_ok=True)
    
print("Training started...")
#length_tr = len(tr_ready)
#length_tes = len(vld_ready)
num_batches = len(train_data_loader)
for e in tqdm(range(cfg.epochs)):        
    act_loss = 0
    if cfg.loss == "JSA":
        for it in range(cfg.num_samples):            
            globals()["prior_samp" + str(it+1)] = bnn_utils.sample_gpriors(shapes, cfg.sigma_pr,ctx)
    
    for i, (data, label) in enumerate(train_data_loader):#        
        label = label.as_in_context(ctx)
        data = bnn_utils.resize_data(data)
        data = data.as_in_context(ctx)        

        with autograd.record():
            # generate sample
            #layer_params, sigmas = generate_weight_sample(shapes, raw_mus, raw_rhos)
            for it in range(cfg.num_samples):
              globals()["epsilons" + str(it+1)] = bnn_utils.sample_epsilons(shapes,ctx)

            # compute softpus for variance
            sigmas = bnn_utils.transform_rhos(raw_rhos) 

            # obtain a sample from q(w|theta) by transforming the epsilons
            for it in range(cfg.num_samples):
              globals()["layer_params" + str(it+1)] = bnn_utils.transform_gaussian_samples(raw_mus, sigmas, globals()["epsilons" + str(it+1)])
              del globals()["epsilons" + str(it+1)]
            

            # overwrite network parameters with sampled parameters
            for ite in range(cfg.num_samples):
              for sample, param in zip(globals()["layer_params" + str(ite+1)], net.collect_params().values()):
                  param._data[0] = sample

              # forward-propagate the batch
              globals()["output" + str(ite+1)] = net(data)
            sz = data.shape[0]
            del data
            
            loss1 = 0
            # calculate the loss
            for it in range(cfg.num_samples): 
              if cfg.loss == "JSA":
                  loss1 = loss1 + bbb_loss(globals()["output" + str(it+1)], label,\
                      globals()["layer_params" + str(it+1)], raw_mus, sigmas,\
                      cfg.lamda, cfg.alpha, globals()["prior_samp" + str(it+1)], nd.array([cfg.sigma_pr], ctx=ctx), num_batches)
              else:
                  loss1 = loss1 +   bbb_loss(globals()["output" + str(it+1)], label, raw_mus, sigmas,cfg.lamda, cfg.alpha,cfg.sigma_pr, num_batches)
            
            for it in range(cfg.num_samples):
                del globals()["output" + str(it+1)], globals()["layer_params" + str(it+1)]
              
            del sigmas, label
            #ctx.empty_cache()
            # backpropagate for gradient calculation
            
            loss1.backward()
            
        trainer.step(sz)
        

        # calculate moving loss for monitoring convergence
        curr_loss = nd.mean(loss1)
        act_loss += nd.sum(loss1)
        #input("mov loss...")
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        #mx.nd.waitall()

        #######################
        #######################
    #mx.nd.waitall()
    #input("Accuracy...")
    vld_accuracy = bnn_metrics.evaluate_accuracy(val_data_loader, net, raw_mus, ctx)
    train_accuracy = bnn_metrics.evaluate_accuracy(train_data_loader, net, raw_mus, ctx)
    print("\n Epoch %s. Loss: %s, Train_acc %s, valid_acc %s " %
          (e, (act_loss/train_size).asscalar(), train_accuracy.asscalar(), vld_accuracy.asscalar()),file=open(cfg.out_dir+'/output.txt','a'))
    #print("\n Epoch %s. Loss: %s, Train_acc %s, valid_acc %s " %
    #      (e, moving_loss.asscalar(), train_accuracy.asscalar(), vld_accuracy.asscalar()),file=open(cfg.out_dir+'/output.txt','a'))

    loss_list.append(act_loss/train_size)
    mov_loss.append(moving_loss)
    train_acc_list.append(train_accuracy)
    vld_acc_list.append(vld_accuracy)
    mx.nd.save(cfg.out_dir+'/loss_bcnn', loss_list)
    mx.nd.save(cfg.out_dir+'/mov_loss_bcnn', mov_loss)
    np.save(cfg.out_dir+'/train_accuracy_bcnn', np.array(train_acc_list))
    np.save(cfg.out_dir+'/vld_accuracy_bcnn',np.array(vld_acc_list))
    mx.nd.save(cfg.out_dir+'/par_mu', raw_mus)
    mx.nd.save(cfg.out_dir+'/par_rho', raw_rhos)
    if vld_accuracy>max_acc:
       max_acc = vld_accuracy
       test_accuracy = bnn_metrics.evaluate_accuracy(test_data_loader,net,raw_mus,ctx)
       print('Max validation accuracy reached. Test accuracy is  >> ',test_accuracy.asscalar(),file=open(cfg.out_dir+'/output.txt','a'))
       mx.nd.save(cfg.out_dir+'/max_par_mu', raw_mus)
       mx.nd.save(cfg.out_dir+'/max_par_rho', raw_rhos)

print("Training completed")

        
