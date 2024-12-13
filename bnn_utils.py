#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function definitions

"""
from mxnet import nd, image
import numpy as np

def sample_epsilons(param_shapes,ctx):
    epsilons = [nd.random_normal(shape=shape, loc=0., scale=1.0, ctx=ctx) for shape in param_shapes]
    return epsilons

def softplus(x):
    return nd.log(1. + nd.exp(x))

def transform_rhos(rhos):
    return [softplus(rho) for rho in rhos]

def transform_gaussian_samples(mus, sigmas, epsilons):
    samples = []
    for j in range(len(mus)):
        samples.append(mus[j] + sigmas[j] * epsilons[j])
    return samples

def generate_weight_sample(layer_param_shapes, mus, rhos):
    # sample epsilons from standard normal
    epsilons = sample_epsilons(layer_param_shapes)

    # compute softplus for variance
    sigmas = transform_rhos(rhos)

    # obtain a sample from q(w|theta) by transforming the epsilons
    layer_params = transform_gaussian_samples(mus, sigmas, epsilons)

    return layer_params, sigmas

def uq_predict_net(feats,lay_params,net):
  for l_param, param in zip(lay_params, net.collect_params().values()):
    param._data[0] = l_param
  return(net(feats))
  
def uq_diagonal_mat(x):
  size = len(x)
  dmat = np.zeros((size,size))
  for i in range(size):
    dmat[i,i] = x[i]
  return dmat

def resize_data(data):
   tmp_train_ft= np.rollaxis(data,1,4)
   for i in range(len(tmp_train_ft)):
     if i==0:        
       tr = image.imresize(nd.array(tmp_train_ft[i]),224,224)
       trn = nd.reshape(tr,(1,tr.shape[0],tr.shape[1],tr.shape[2]))
     else:
       tr = image.imresize(nd.array(tmp_train_ft[i]),224,224) 
       trn = nd.concat(trn,nd.reshape(tr,(1,tr.shape[0],tr.shape[1],tr.shape[2])), dim=0) 
   tmp_train_feat= np.rollaxis(trn,3,1)
   #del trn
   tmp_tr_ready = tmp_train_feat
   #del tmp_train_feat, tmp_train_ft
   return tmp_tr_ready

def sample_gpriors(param_shapes, sigma, ctx):
    epsilons = [nd.random_normal(
        shape=shape, loc=0., scale=sigma, ctx=ctx) for shape in param_shapes]
    return epsilons
