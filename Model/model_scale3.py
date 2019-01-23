# coding: utf-8

# In[1]:


import math
import tensorflow as tf
import numpy as np
import os
import scipy.stats
import time
# from crfrnn_layer_1 import CrfRnnLayer


# In[2]:


FLAGS=tf.app.flags.FLAGS


# In[3]:


def conv2d( x, n_in, n_out, k, s, p='SAME', bias=False, scope='conv'):
    with tf.variable_scope(scope):
        kernel=tf.Variable(tf.truncated_normal([ k, k, n_in, n_out], stddev=math.sqrt(2/(k*k*n_in))),name='weight')
        tf.add_to_collection('weight',kernel)
        conv=tf.nn.conv2d(x,kernel,[1,s,s,1],padding=p)
        if bias:
            bias=tf.get_variable('bais',[n_out],initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('bias',bias)
            conv=tf.nn.bias_add(conv,bias)
    return conv


# In[4]:

def batch_norm(x, n_out, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta')
        gamma = tf.Variable(tf.truncated_normal([n_out], stddev=0.1), name='gamma')
        tf.add_to_collection('biases', beta)
        tf.add_to_collection('weights', gamma)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        normed = tf.nn.batch_norm_with_global_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3,
                                                            scale_after_normalization=True)
    return normed

def deconv_layer(x,output_shape, n_in, n_out, k, p='SAME', bias=False, scope='deconv'):
    with tf.variable_scope(scope):
        kernel=tf.Variable(tf.truncated_normal([ k, k, n_out, n_in], stddev=math.sqrt(2/(k*k*n_in))),name='weight')
        tf.add_to_collection('weight',kernel)
        deconv=tf.nn.conv2d_transpose(x,kernel,output_shape,[1,2,2,1],padding=p)
        if bias:
            bias=tf.get_variable('bais',[n_out],initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('bias',bias)
            deconv=tf.nn.bias_add(deconv,bias)
    return deconv



# In[6]:


def inference_1( x, scope='model'):
    with tf.variable_scope(scope):
        num = 4
        ##scale1____________________________________________________________________gloable coarse_scale network
        #input  160*576*1
        #output 20*72*96
        coarse1 = conv2d(x, 1, 96, 11, 4, 'SAME', True, scope='coarse1')   # 1
        coarse1_bn = batch_norm(coarse1, 96, scope='coarse1_bn')
        coarse1_relu = tf.nn.relu(coarse1_bn, name='relu_coarse1')
        pool1  = tf.nn.max_pool(coarse1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')
        #input  20*72*96
        #output 10*36*256
        coarse2 = conv2d(pool1, 96, 256, 5, 1, 'SAME', True, scope='coarse2')  # 2
        coarse2_bn = batch_norm(coarse2, 256, scope='coarse2_bn')
        coarse2_relu = tf.nn.relu(coarse2_bn, name='coarse2_relu')
        pool2 = tf.nn.max_pool(coarse2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool2')

        #input  10*36*256
        #output 10*36*384
        coarse3 = conv2d(pool2, 256, 384, 3, 1, 'SAME', True, scope='coarse3')  # 3
        coarse3_bn = batch_norm(coarse3, 384, scope='coarse3_bn')
        coarse3_relu = tf.nn.relu(coarse3_bn, name='coarse3_relu')

        #input  10*36*384
        #output 10*36*384
        coarse4 = conv2d(coarse3_relu, 384, 384, 3, 1, 'SAME', True, scope='coarse4') # 4
        coarse4_bn = batch_norm(coarse4, 384, scope='coarse4_bn')
        coarse4_relu = tf.nn.relu(coarse4_bn, name='coarse4_relu')

        #input  10*36*384
        #output 10*36*256
        coarse5 = conv2d(coarse4_relu, 384, 256, 3, 1, 'SAME', True, scope='coarse5')  # 5
        coarse5_bn = batch_norm(coarse5, 256, scope='coarse5_bn')
        coarse5_relu = tf.nn.relu(coarse5_bn, name='coarse5_relu')
        
        #input  10*36*256
        #output 20*72*256
        with tf.name_scope('deconv5_image') as scope:
            wt5=tf.Variable(tf.truncated_normal([3,3,256,256]))
            deconv5_image=tf.nn.relu(batch_norm(tf.nn.conv2d_transpose(coarse5_relu,wt5,[num,20,72,256],[1,2,2,1],padding='SAME'),256))
        #input  20*72*256
        #output 40*144*256
        with tf.name_scope('deconv6_image') as scope:
            wt6=tf.Variable(tf.truncated_normal([3,3,256,256]))
            deconv6_image=tf.nn.relu(batch_norm(tf.nn.conv2d_transpose(deconv5_image,wt6,[num,40,144,256],[1,2,2,1],padding='SAME'),256))
        
        fc6=conv2d(deconv6_image,256,32,1,1,'SAME',True,scope='fc6')   # 6
        fc6_bn = batch_norm(fc6, 32, scope='fc6_bn')
        coarse6=tf.nn.relu(fc6_bn,name='coarse6')
        fc7=conv2d(coarse6,32,1,1,1,'SAME',True,scope='fc7')    # 7
        fc7_bn = batch_norm(fc7, 1, scope='fc7_bn')
        coarse7_output=tf.nn.relu(fc7_bn,name='coarse7_output')
    
        ###scale2___________________________________________________________________local fine_scale network
        #input  160*576*1
        #output 40*144*63
        fine1_conv = conv2d(x, 1, 63, 9, 2, 'SAME', True, scope='fine1')    # 8
        fine1_bn = batch_norm(fine1_conv, 63, scope='fine1_bn')
        fine1_relu = tf.nn.relu(fine1_conv,name='fine1_relu')
        fine1 = tf.nn.max_pool(fine1_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
                                padding='SAME', name='fine_pool1')
        fine1_dropout = tf.nn.dropout(fine1, 0.8)
        
        #input  40*144*63
        #output 40*144*64
        fine2 = tf.concat([fine1_dropout, coarse7_output],3)
        fine3 = conv2d( fine2, 64, 64, 5, 1, 'SAME', True, scope='fine3')   # 9
        fine3_bn = batch_norm(fine3, 64, scope='fine3_bn')
        fine3_relu = tf.nn.relu(fine3_bn,name='fine3_relu')
        fine3_dropout = tf.nn.dropout(fine3_relu, 0.8)
        fine4 = conv2d(fine3_dropout, 64, 64, 5, 1, 'SAME', True, scope='fine4')  # 10
        fine4_bn = batch_norm(fine4, 64, scope='fine4_bn')
        fine4_relu = tf.nn.relu(fine4_bn,name='fine4_relu')

        ###scale3______________________________________________________________________ Higher Resolution 
        #input  160*576*1
        #output 40*144*96
        scale3_conv1 = conv2d(x, 1, 96, 9, 2, 'SAME', True, scope='scale3_conv1')   # 11
        scale3_bn1 = batch_norm(scale3_conv1, 96, scope='scale3_bn1')
        scale3_relu = tf.nn.relu(scale3_bn1,name='scale3_relu')
        
        scale3_pool1 = tf.nn.max_pool(scale3_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
                                      padding='SAME', name='scale3_pool1')
        
        #input  40*144*96
        #output 40*144*64+96=160
        scale3_1 = tf.concat([scale3_pool1, fine4_relu],3)
        #input  40*144*64+96=160
        #output 40*144*64
        scale3_2 = conv2d(scale3_1, 160, 64, 5, 1, 'SAME', True, scope='scale3_2')   # 12
        scale3_bn2 = batch_norm(scale3_2, 64, scope='scale3_bn2')
        scale3_2_relu = tf.nn.relu(scale3_bn2,name='scale3_2_relu')
        #input  40*144*64
        #output 16*40*144*32
        scale3_3 = conv2d(scale3_2_relu, 64, 64, 5, 1, 'SAME', True, scope='scale3_3')   # 13
        scale3_bn3 = batch_norm(scale3_3, 64, scope='scale3_bn3')
        scale3_3_relu = tf.nn.relu(scale3_3,name='scale3_3_relu')
        scale3_4 = conv2d(scale3_3, 64, 32, 5, 1, 'SAME', True, scope='scale3_4')    # 14
        scale3_bn4 = batch_norm(scale3_4, 32, scope='scale3_bn4')
        # scale3_4_relu = tf.nn.relu(scale3_bn4,name='scale3_4_relu')
        #middile = tf.reshape(scale3_4_relu[0], [1, 40,144])

        print('model has been done...')
      
    return  scale3_bn4




# In[7]:


def loss(pre_dep,true_dep,beta1,beta2,beta3,lamda):
    
    # define unary loss function
    dim=pre_dep.get_shape()[3].value
    logits=tf.reshape(pre_dep,[-1,dim])     # re_dep.shape: (23040, 32)    4*40*144=23040  batch_size*40*144
    labels=tf.reshape(true_dep,[-1])        # ground_truth.shape: (23040,)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy')
    cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy_mean')   # 所有元素求平均值
    tf.add_to_collection('losses',cross_entropy_mean)
    print('unary-loss- function-done...')

    # pairwise loss function
    time_start=time.time()
    num = 4      # num 指的是batch_szie
    pred = tf.argmax(pre_dep, 3)       # 输出一张depth map(batchsize=4不会变)
    feature = tf.cast(tf.reshape(pred,[num,40,144]),tf.float32)
    label = tf.cast(tf.reshape(true_dep,[num,40,144]),tf.float32)
    
    ################## 尺度 1 
    feature_pad = tf.pad(feature ,[[0,0],[1,1],[1,1]],"REFLECT")   # 三个维度：batch_size,height,width.对高和宽分别补（对称补非0操作）
    label_pad = tf.pad(label ,[[0,0],[1,1],[1,1]],"REFLECT")
    feature_new = tf.Variable(tf.zeros([num,40*144,9]))   # 存放特征矩阵,8邻域，针对图片的每个pixel都做了邻域操作，那么一张图片就要做40*144
    label_new = tf.Variable(tf.zeros([num,40*144,9]))     # 存放标签矩阵
    for i in range (1,40+1):
        for j in range (1,144+1):
            index = (i-1)*144 + j-1
            feature_patch = feature_pad[:, i-1:i+2, j-1:j+2] # 取出8邻域元素，有9个
            label_patch = label_pad[:, i-1:i+2, j-1:j+2]
            feature_col = tf.reshape(feature_patch,[-1,9]) 
            label_col = tf.reshape(label_patch,[-1,9])
            feature_new[:,index,:].assign(feature_col)
            label_new[:,index,:].assign(label_col)
    feature_mean = tf.reduce_mean(feature_new,2)      # 计算第三个维度的均值  [num,40*144]
    label_mean = tf.reduce_mean(label_new,2)
    feature_new = feature_new[:,:,4:5] - feature_new  # 中心元素减去四周的元素（fi-fj）
    label_new = label_new[:,:,4:5] - label_new        # 中心元素减去四周的元素 （di-dj）
    print('scale1_loss has been done...')   #   ######   feature_new,label_new----sacle1

    feature_mean = tf.reshape(feature_mean,[num,40,144])
    label_mean = tf.reshape(label_mean,[num,40,144])


    ############### 尺度2
    feature_pad_scale2 = tf.pad(feature_mean ,[[0,0],[1,1],[1,1]],"REFLECT")   # 三个维度：batch_size,height,width.对高和宽分别补（对称补非0操作）
    label_pad_scale2 = tf.pad(label_mean ,[[0,0],[1,1],[1,1]],"REFLECT")
    feature_scale2 = tf.Variable(tf.zeros([num,40*144,9]))
    label_scale2 = tf.Variable(tf.zeros([num,40*144,9]))
    for p in range (1,40+1):
        for q in range (1,144+1):
            index_scale2 = (p-1)*144 + q-1
            feature_patch_scale2 = feature_pad_scale2[:, p-1:p+2, q-1:q+2] # 取出8邻域元素，有9个
            label_patch_scale2 = label_pad_scale2[:, p-1:p+2, q-1:q+2]
            feature_col_scale2 = tf.reshape(feature_patch_scale2,[-1,9]) 
            label_col_scale2 = tf.reshape(label_patch_scale2,[-1,9])
            feature_scale2[:,index_scale2,:].assign(feature_col_scale2)
            label_scale2[:,index_scale2,:].assign(label_col_scale2)    
    feature_mean_scale3 = tf.reduce_mean(feature_scale2,2)      # 计算第三个维度的均值  [num,40*144]
    label_mean_scale3 = tf.reduce_mean(label_scale2,2)
    feature_scale2 = feature_scale2[:,:,4:5] - feature_scale2  # 中心元素减去四周的元素（fi-fj）
    label_scale2 = label_scale2[:,:,4:5] - label_scale2        # 中心元素减去四周的元素 （di-dj）

    feature_mean_scale3 = tf.reshape(feature_mean_scale3,[num,40,144])
    label_mean_scale3 = tf.reshape(label_mean_scale3,[num,40,144])

    ############### 尺度3
    feature_pad_scale3 = tf.pad(feature_mean_scale3 ,[[0,0],[1,1],[1,1]],"REFLECT")   # 三个维度：batch_size,height,width.对高和宽分别补（对称补非0操作）
    label_pad_scale3 = tf.pad(label_mean_scale3 ,[[0,0],[1,1],[1,1]],"REFLECT")
    # print ('feature_mean的大小',feature_mean.shape)   # feature_mean的大小 (4, 5760)
    feature_scale3 = tf.Variable(tf.zeros([num,40*144,9]))
    label_scale3 = tf.Variable(tf.zeros([num,40*144,9]))
    for m in range (1,40+1):
        for n in range (1,144+1):
            index_scale3 = (m-1)*144 + n-1
            feature_patch_scale3 = feature_pad_scale3[:, m-1:m+2, n-1:n+2] # 取出8邻域元素，有9个
            label_patch_scale3 = label_pad_scale3[:, m-1:m+2, n-1:n+2]
            feature_col_scale3 = tf.reshape(feature_patch_scale3,[-1,9]) 
            label_col_scale3 = tf.reshape(label_patch_scale3,[-1,9])
            feature_scale3[:,index_scale3,:].assign(feature_col_scale3)
            label_scale3[:,index_scale3,:].assign(label_col_scale3)
    feature_scale3 = feature_scale3[:,:,4:5] - feature_scale3  # 中心元素减去四周的元素（fi-fj）
    label_scale3 = label_scale3[:,:,4:5] - label_scale3        # 中心元素减去四周的元素 （di-dj）
    print('scale3_loss has been done...')
    

    ##### 计算loss
    # pairwise_loss.shape = [16,5760,9]，计算了16(batch_size)张图片的损失
    # tf.multiply()----element_wise
    pairwise_loss_sacale1 = lamda*tf.multiply(tf.exp(-beta1* tf.square(feature_new)), tf.square(label_new))  # lamda*exp(-beta*(fi-fj)^2)*(di-dj)^2
    pairwise_loss_sacale2 = lamda*tf.multiply(tf.exp(-beta2* tf.square(feature_scale2)), tf.square(label_scale2))
    pairwise_loss_sacale3 = lamda*tf.multiply(tf.exp(-beta3* tf.square(feature_scale3)), tf.square(label_scale3))
    pairwise_loss = (pairwise_loss_sacale1 + pairwise_loss_sacale2 + pairwise_loss_sacale3)/3
    pairwise_loss = tf.reshape(pairwise_loss,[-1,1])
    pairwise_loss_mean = 9*tf.reduce_mean(pairwise_loss,name='pairwise_loss_mean')  # 所有元素求平均值
    tf.add_to_collection('losses',pairwise_loss_mean)
    print('pairwise-loss- function-done...')
    time_end=time.time()
    print('pairwise time cost',time_end-time_start,'s')   # pairwise time cost 49.375667333602905 s



    weight_l2_losses=[tf.nn.l2_loss(o) for o in tf.get_collection('weight')]
    # print('weight_l2_losses:-------------------',weight_l2_losses[10]) # weight_l2_losses:Tensor("L2Loss_10:0", shape=(), dtype=float32)
    weight_decay_loss=tf.multiply(1e-4,tf.add_n(weight_l2_losses),name='weight_decay_loss')
    tf.add_to_collection('losses',weight_decay_loss)
    print('L2-loss- function-done...')
    return tf.add_n(tf.get_collection('losses'),name='total_loss')


# In[8]:


def train(loss,global_step):
    optimizer=tf.train.AdamOptimizer(FLAGS.lr)
    train_op=optimizer.minimize(loss,global_step=global_step)
    return train_op
