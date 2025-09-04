# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

def kronecker_matmul(x, hadL, hadR):
    """equivalent to
    
        had = torch.kron(hadL, hadR)
        x = x.reshape(-1, had.shape[0])
        x = x.matmul(had).reshape(init_shape)
    """
    init_shape = x.shape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    return x.reshape(init_shape)


class PerChannelScaling(nn.Module):
    """Per-channel scaling module adapted from FlatQuant"""
    
    def __init__(self, size, add_diag=True, diag_init_para=None):
        super(PerChannelScaling, self).__init__()
        self.add_diag = add_diag
        self.use_diag = True
        
        if self.add_diag:
            if diag_init_para is None:
                self.diag_scale = nn.Parameter(torch.ones(size, dtype=torch.float32), requires_grad=True)
            else:
                self.diag_scale = nn.Parameter(diag_init_para, requires_grad=True)
        
    def forward(self, inp, inv_t=False):
        if self.add_diag and self.use_diag:
            if inv_t:
                return inp / self.diag_scale.to(inp)
            else:
                return inp * self.diag_scale.to(inp)
        return inp
    
    def reparameterize(self):
        """Set use_diag to False to disable scaling"""
        self.use_diag = False


class TransformationMatrix(nn.Module):
    """Transformation matrix with per-channel scaling adapted from FlatQuant"""
    
    def __init__(self, left_size, right_size, add_diag=False, diag_init_para=None):
        super(TransformationMatrix, self).__init__()
        
        # Initialize transformation matrices
        self.linear_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_left.weight.data = torch.eye(left_size, dtype=torch.float32)
        
        self.linear_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_right.weight.data = torch.eye(right_size, dtype=torch.float32)
        
        # Per-channel scaling
        self.add_diag = add_diag
        self.use_diag = True
        if self.add_diag:
            if diag_init_para is None:
                self.diag_scale = nn.Parameter(torch.ones(left_size * right_size, dtype=torch.float32), requires_grad=True)
            else:
                self.diag_scale = nn.Parameter(diag_init_para, requires_grad=True)
        
        self._eval_mode = False
    
    def forward(self, inp, inv_t=False):
        # Apply per-channel scaling
        if self.add_diag and self.use_diag:
            if inv_t:
                inp = inp / self.diag_scale.to(inp)
            else:
                inp = inp * self.diag_scale.to(inp)
        
        # Apply transformation matrices
        if not self._eval_mode:
            matrix_left, matrix_right = self.linear_left.weight, self.linear_right.weight
            if inv_t:
                matrix_left = torch.inverse(matrix_left).T
                matrix_right = torch.inverse(matrix_right).T
        else:
            matrix_left, matrix_right = self.matrix_left, self.matrix_right
            if inv_t:
                matrix_left, matrix_right = self.matrix_left_inv, self.matrix_right_inv
        
        return kronecker_matmul(inp, matrix_left.to(inp), matrix_right.to(inp))
    
    def to_eval_mode(self):
        if not self._eval_mode:
            self.matrix_left = nn.Parameter(self.linear_left.weight, requires_grad=False)
            self.matrix_right = nn.Parameter(self.linear_right.weight, requires_grad=False)
            self.matrix_left_inv = nn.Parameter(torch.inverse(self.linear_left.weight).T, requires_grad=False)
            self.matrix_right_inv = nn.Parameter(torch.inverse(self.linear_right.weight).T, requires_grad=False)
            del self.linear_left, self.linear_right
            self._eval_mode = True
    
    def reparameterize(self):
        """Set use_diag to False to disable scaling"""
        self.use_diag = False