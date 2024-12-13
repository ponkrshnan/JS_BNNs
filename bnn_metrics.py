#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions and their associated dependencies

"""
from mxnet import gluon, nd
import numpy as np
import bnn_utils

class BBBLoss_JSG(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BBBLoss_JSG, self).__init__(weight, batch_axis, **kwargs)


    def hybrid_forward(self, F, output, label, mus, sigmas, lamda, alpha, sigma_pr, num_batches, sample_weight=None):
        log_likelihood_sum = nd.sum(-nd.softmax_cross_entropy(output, label))
        sum_js = 0
        mu_pr = 0
        for mu,sigma in zip(mus,sigmas):
            sigma_alp = 1/(alpha/sigma + (1-alpha)/sigma_pr)
            mu_alp = sigma_alp*(alpha * mu/sigma +(1- alpha) * mu_pr/sigma_pr)
            js_loss = 0.5*(((1-alpha)*sigma+ alpha*sigma_pr)/sigma_alp + nd.log(sigma_alp/(sigma**(1-alpha) * sigma_pr**alpha)) + (1-alpha)* (mu_alp - mu)**2 / sigma_alp + alpha* (mu_alp-mu_pr)**2 /sigma_alp -1)
            sum_js = sum_js + nd.sum(js_loss)
        loss = (lamda*sum_js)/num_batches - log_likelihood_sum
        return loss

class BBBLoss_JSA(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BBBLoss_JSA, self).__init__(weight, batch_axis, **kwargs)

    def log_gaussian(self, x, mu, sigma):
        return -0.5 * np.log(2.0 * np.pi) - nd.log(sigma+1e-7) - (x - mu) ** 2 / (2 * (sigma+1e-7) ** 2)

    def gaussian(self, x, mu, sigma):
        scaling = 1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))
        bell = nd.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
        return scaling * bell

    def arth_mean(self, x, qmu, qsigma,	psigma, alpha):
        q = self.gaussian(x, qmu, qsigma)
        p = self.gaussian(x, 0., psigma)
        m = alpha*q + (1.0-alpha)*p
        return nd.log(m)

    def hybrid_forward(self, F, output, label, params, mus, sigmas, lamda, alpha, prior_samples, psigma, num_batches, sample_weight=None):
#        psigma = nd.array([psigma], ctx=ctx)
        log_likelihood_sum = nd.sum(-nd.softmax_cross_entropy(output, label))
        post_log_var_post_sum = sum([nd.sum(self.log_gaussian(params[i], mus[i], sigmas[i])) for i in range(len(params))])
        post_log_arthmean_sum = sum([nd.sum(self.arth_mean(params[i], mus[i], sigmas[i], psigma, alpha)) for i in range(len(params))])
        prior_log_prior_sum = sum([nd.sum(self.log_gaussian(j,0.,psigma)) for j in prior_samples])
        prior_log_arthmean_sum = sum([nd.sum(self.arth_mean(prior_samples[i], mus[i], sigmas[i], psigma, alpha)) for i in range(len(params))])
        loss = 1.0 / (num_batches) * lamda* (1-alpha) *(post_log_var_post_sum - post_log_arthmean_sum) + \
        1.0 / (num_batches) * lamda* alpha *(prior_log_prior_sum - prior_log_arthmean_sum) - log_likelihood_sum
        return loss  

class BBBLoss_KL(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BBBLoss_KL, self).__init__(weight, batch_axis, **kwargs)

#	lamda and alpha are dummy variables unused here
    def hybrid_forward(self, F, output, label, mus, sigmas, lamda, alpha, sigma_pr,  num_batches, sample_weight=None): 
        log_likelihood_sum = nd.sum(-nd.softmax_cross_entropy(output, label))
        sum_kl = 0
        mu_pr = 0
        for mu,sigma in zip(mus,sigmas):  
            kl_loss = 0.5*(sigma/sigma_pr + nd.log(sigma_pr/sigma) + (mu_pr - mu)**2 / sigma_pr -1)
            sum_kl = sum_kl + nd.sum(kl_loss)
        loss = sum_kl/num_batches - log_likelihood_sum
#        div = sum_kl/num_batches
#        like =  log_likelihood_sum
        return loss
    
def evaluate_accuracy(data_iterator, net, layer_params, ctx):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        label = label.as_in_context(ctx)
        data = bnn_utils.resize_data(data)
        data = data.as_in_context(ctx)             
        for l_param, param in zip(layer_params, net.collect_params().values()):
            param._data[0] = l_param
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
        #nd.waitall()
    return (numerator / denominator)
