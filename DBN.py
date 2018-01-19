#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 18:01:00 2018

@author: bizzy
"""

# For Drawing graphs
import matplotlib.pyplot as plt
# For importing MNIST data
import input_data
from RBM_class import RBM

# Import MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX = mnist.train.images

n_v   = 784       
n_h1  = 500       
n_h2  = 250
n_h3  = 120
n_h4  = 60
n_h5  = 30
n_h6  = 25
n_h7  = 20
n_h8  = 15
n_h9  = 10
n_h10 = 5

n_batch = 20
epochs = 2000

# Layer 1
R1 = RBM(n_v, n_h1, n_batch, epochs, 0.1, trX)
R1.CD(1); R1.Save_data("a1","b1","w1");

# Layer 2
R2_input = R1.Propagation(trX, R1.w, R1.b)
R2 = RBM(n_h1, n_h2, n_batch, epochs, 0.1, R2_input)
R2.CD(2); R2.Save_data("a2","b2","w2");

# Layer 3
R3_input = R2.Propagation(R2_input, R2.w, R2.b) 
R3 = RBM(n_h2, n_h3, n_batch, epochs, 0.1, R3_input)
R3.CD(3); R3.Save_data("a3","b3","w3");

# Layer 4
R4_input = R3.Propagation(R3_input, R3.w, R3.b) 
R4 = RBM(n_h3, n_h4, n_batch, epochs, 0.1, R4_input)
R4.CD(4); R4.Save_data("a4","b4","w4");

# Layer 5
R5_input = R4.Propagation(R4_input, R4.w, R4.b) 
R5 = RBM(n_h4, n_h5, n_batch, epochs, 0.1, R5_input)
R5.CD(5); R5.Save_data("a5","b5","w5");

# Layer 6
R6_input = R5.Propagation(R5_input, R5.w, R5.b)
R6 = RBM(n_h5, n_h6, n_batch, epochs, 0.1, R6_input)
R6.CD(6); R6.Save_data("a6","b6","w6");

# Layer 7
R7_input = R6.Propagation(R6_input, R6.w, R6.b)
R7 = RBM(n_h6, n_h7, n_batch, epochs, 0.1, R7_input)
R7.CD(7); R7.Save_data("a7","b7","w7");

# Layer 8
R8_input = R7.Propagation(R7_input, R7.w, R7.b)
R8 = RBM(n_h7, n_h8, n_batch, epochs, 0.1, R8_input)
R8.CD(8); R8.Save_data("a8","b8","w8");

# Layer 9
R9_input = R8.Propagation(R8_input, R8.w, R8.b)
R9 = RBM(n_h8, n_h9, n_batch, epochs, 0.1, R9_input)
R9.CD(9); R9.Save_data("a9","b9","w9");

# Layer 10
R10_input = R9.Propagation(R9_input, R9.w, R9.b)
R10 = RBM(n_h9, n_h10, n_batch, epochs, 0.1, R10_input)
R10.CD(10); R10.Save_data("a10","b10","w10");




'''
r = rd.randint(55000)
plt.subplot(4,4,1); plt.imshow(trX[r].reshape(28,28));

old_v = trX[r]
for i in range(15):
    new_h = R.Sigmoid(np.dot(old_v.reshape(1, n_v), w) + b)
    new_v = R.Sigmoid(np.dot(new_h, w.T) + a)
    plt.subplot(4,4,2+i); plt.imshow(new_v.reshape(28,28));
    old_v = new_v
'''