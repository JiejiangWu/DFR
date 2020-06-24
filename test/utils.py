#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:50:04 2019

@author: magic
"""
import torch
import numpy as np


def save_checkpoint(path, iteration, iterations_since_improvement, model,optimizer,
                    best_metric, recent_metric, Losses_during_iteration, metric_during_iteration,is_best):
    state = {'iteration': iteration,
             'iterations_since_improvement': iterations_since_improvement,
             'best_metric': best_metric,
             'recent_metric': recent_metric,
             'model': model,
             'optimizer':optimizer,
             'Losses_during_iteration':Losses_during_iteration,
             'metric_during_iteration':metric_during_iteration}
    
    filename = path + '/Temp.pth.tar'
    best_filename = path + '/Best.pth.tar'
    torch.save(state, filename)
#     If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, best_filename)    
        
def save_named_checkpoint(name,path, iteration, iterations_since_improvement, model,optimizer,
                    best_metric, recent_metric, Losses_during_iteration, metric_during_iteration,is_best):
    state = {'iteration': iteration,
             'iterations_since_improvement': iterations_since_improvement,
             'best_metric': best_metric,
             'recent_metric': recent_metric,
             'model': model,
             'optimizer':optimizer,
             'Losses_during_iteration':Losses_during_iteration,
             'metric_during_iteration':metric_during_iteration}
    
    filename = path + '/' + name + '_Temp.pth.tar'
    best_filename = path + '/' + name + '_Best.pth.tar'
    torch.save(state, filename)
#     If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, best_filename)    

#def save_checkpoint(path, epoch, epochs_since_improvement, model,optimizer,
#                    best_metric, recent_metric, Losses_during_iteration, metric_during_iteration,is_best):
#    state = {'epoch': epoch,
#             'epochs_since_improvement': epochs_since_improvement,
#             'best_metric': best_metric,
#             'recent_metric': recent_metric,
#             'model': model,
#             'optimizer':optimizer,
#             'Losses_during_iteration':Losses_during_iteration,
#             'metric_during_iteration':metric_during_iteration}
#    
#    filename = path + '/Temp.pth.tar'
#    best_filename = path + '/Best.pth.tar'
#    torch.save(state, filename)
##     If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
#    if is_best:
#        torch.save(state, best_filename)    
#
#def save_named_checkpoint(name,path, epoch, epochs_since_improvement, model,optimizer,
#                    best_CD, recent_CD, Losses_during_epoch, CD_during_epoch,is_best):
#    state = {'epoch': epoch,
#             'epochs_since_improvement': epochs_since_improvement,
#             'best_CD': best_CD,
#             'recent_CD': recent_CD,
#             'model': model,
#             'optimizer':optimizer,
#             'Losses_during_epoch':Losses_during_epoch,
#             'CD_during_epoch':CD_during_epoch}
#    
#    filename = path + '/' + name + '.pth.tar'
#    filename2 = path + '/Temp.pth.tar'
#    best_filename = path + '/Best.pth.tar'
#    torch.save(state, filename)
#    torch.save(state, filename2)
##     If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
#    if is_best:
#        torch.save(state, best_filename)    

    
def write_obj(path,vertices,faces):
    with open(path, 'w') as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v %.4f %.4f %.4f\n" % (v[0],v[1],v[2]))
        for p in faces:
            f.write("f")
            for i in p:
                f.write(" %d" % (i + 1))
            f.write("\n")

def write_points(path,v):
    np.savetxt(path,v)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count