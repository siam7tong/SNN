# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
# import timeit
# import pickle
import cPickle
import os
# import datetime
# import cv2
import lasagne
# import random
import matplotlib
# from numpy import dtype
# from collections import OrderedDict
import matplotlib.pyplot as plt


# from theano.compile.nanguardmode import NanGuardMode
# from calface_reader import data_set, mnist_data_set
# from mnist_reader import data_set, mnist_data_set

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
# from lasagne.layers import NonlinearityLayer
# from lasagne.layers import DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
# from lasagne.layers import GlobalPoolLayer as GapLayer
# from lasagne.nonlinearities import softmax, sigmoid
# from lasagne.layers import ElemwiseSumLayer
# from lasagne.layers import batch_norm
# from lasagne.nonlinearities import rectify
# from visualize import plot_conv_weights,
# import scipy.io as sio
# from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from stdp import stdpOp
from mnist_reader import data_set, mnist_data_set
from itertools import product as iter_product

import sys
reload(sys)
sys.setdefaultencoding('utf8')

matplotlib.use('Agg')
theano.config.floatX = 'float32'
# nvcc.flags = D_FORCE_INLINES


def relu1(x):
    return T.switch(x < 0, 0, x)


def std_conv_layer(input, num_filters, filter_shape, pad='same',
                   nonlinearity=lasagne.nonlinearities.rectify,
                   w=None,
                   # W = lasagne.init.Normal(std = 0.01, mean = 0.0),
                   b=lasagne.init.Constant(0.),
                   do_batch_norm=False):
    if w is None:
        if nonlinearity == lasagne.nonlinearities.rectify:
            print 'convlayer: rectifier func'
            w = lasagne.init.HeNormal(gain='relu')
        else:
            print 'convlayer: sigmoid func'
            w = lasagne.init.HeNormal(1.0)
    else:
        print 'convlayer: W not None'
    conv_layer = ConvLayer(input, num_filters, filter_shape,
                           pad=pad, flip_filters=False,
                           W=w, b=b,
                           nonlinearity=nonlinearity)
    if do_batch_norm:
        conv_layer = lasagne.layers.batch_norm(conv_layer)
    else:
        print 'convlayer: No batch norm.'
    return conv_layer

# 这里当时用不到，就注释掉了
# from lasagne.layers import TransposedConv2DLayer as DeconvLayer
# from lasagne.layers import ExpressionLayer

# try:

# except:
#    from new_conv import TransposedConv2DLayer as DeconvLayer

# try:

# except:
#    from new_special import ExpressionLayer


###############################################################################
W1 = np.float32([[1.96519161e-05, 2.39409349e-04, 1.07295826e-03, 1.76900911e-03,
                1.07295826e-03, 2.39409349e-04, 1.96519161e-05],
                [2.39409349e-04, 2.91660295e-03, 1.30713076e-02, 2.15509428e-02,
                1.30713076e-02, 2.91660295e-03, 2.39409349e-04],
                [1.07295826e-03, 1.30713076e-02, 5.85815363e-02, 9.65846250e-02,
                5.85815363e-02, 1.30713076e-02, 1.07295826e-03],
                [1.76900911e-03, 2.15509428e-02, 9.65846250e-02, 1.59241126e-01,
                9.65846250e-02, 2.15509428e-02, 1.76900911e-03],
                [1.07295826e-03, 1.30713076e-02, 5.85815363e-02, 9.65846250e-02,
                5.85815363e-02, 1.30713076e-02, 1.07295826e-03],
                [2.39409349e-04, 2.91660295e-03, 1.30713076e-02, 2.15509428e-02,
                1.30713076e-02, 2.91660295e-03, 2.39409349e-04],
                [1.96519161e-05, 2.39409349e-04, 1.07295826e-03, 1.76900911e-03,
                1.07295826e-03, 2.39409349e-04, 1.96519161e-05]])

W2 = np.float32([[0.00492233, 0.00919613, 0.01338028, 0.01516185, 0.01338028, 0.00919613,
                0.00492233],
                [0.00919613, 0.01718062, 0.02499766, 0.02832606, 0.02499766, 0.01718062,
                0.00919613],
                [0.01338028, 0.02499766, 0.03637138, 0.04121417, 0.03637138, 0.02499766,
                0.01338028],
                [0.01516185, 0.02832606, 0.04121417, 0.04670178, 0.04121417, 0.02832606,
                0.01516185],
                [0.01338028, 0.02499766, 0.03637138, 0.04121417, 0.03637138, 0.02499766,
                0.01338028],
                [0.00919613, 0.01718062, 0.02499766, 0.02832606, 0.02499766, 0.01718062,
                0.00919613],
                [0.00492233, 0.00919613, 0.01338028, 0.01516185, 0.01338028, 0.00919613,
                0.00492233]])
# W1 W2都是对称矩阵
W = np.stack((W1, W2), axis=0)
# 就是个简单的拼接，0表示按行拼接，1表示按列，负数就是倒数的维度
dog_W = np.reshape(W, (2, 1, 7, 7))
# 之后用作DoG图的conv操作的filter，(output channels, input channels, filter rows, filter columns)

dog_W = dog_W.astype(theano.config.floatX)


###################################################################################


def plot_weights(W, plot_name, file_path='.', max_subplots=100, max_figures=64, figsize=(28, 28)):
    # Matplotlib 里的常用类的包含关系为 Figure -> Axes -> (Line2D, Text, etc.)一个Figure对象可以包含多个子图(Axes)，
    # 在matplotlib中用Axes对象表示一个绘图区域，可以理解为子图。
    try:
        W = W.get_value(borrow=True)
    except:
        W = W
        ''' W=np.reshape(W,(2,5,5,32))
        W=np.swapaxes(W,0,3)
        W=np.swapaxes(W,3,1)
        W=np.swapaxes(W,2,3)'''
    # W = W/4
    shape = W.shape
    assert ((len(shape) == 2) or (len(shape) == 4))
    max_val = np.max(W)
    min_val = np.min(W)

    if len(shape) == 2:
        plt.figure(figsize=figsize)  # 创建一个fig窗口
        plt.imshow(W, cmap='gray', vmax=max_val, vmin=min_val, interpolation='none')  # imshow将数据标准化为最小和最大值。
        # 您可以使用vmin和vmax参数或norm参数来控制
        plt.axis('off')  # 关闭坐标轴
        plt.colorbar()  # 增加颜色类标
        file_name = plot_name + '.png'
        plt.savefig(os.path.join(file_path, file_name))
        plt.clf()
        plt.close()
        return

    ncols = min(np.ceil(np.sqrt(shape[1])).astype(int),
                np.floor(np.sqrt(max_subplots)).astype(int))
    nrows = ncols-1
    '''
    max_val = -np.inf
    min_val = np.inf
    for i in range(shape[0]):
        tmp = np.mean(W[i], axis = 0)
        max_val = max(max_val, np.max(tmp))
        min_val = min(min_val, np.min(tmp))
    '''
    for j in range(min(shape[0], max_figures)):
        figs, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                  squeeze=False)
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        for i, (r, c) in enumerate(iter_product(range(nrows), range(ncols))):  # i是索引
            if i >= shape[1]:  # 30通道数
                break
            im = axes[r, c].imshow(W[j, i], cmap='Greens', vmax=max_val, vmin=min_val, interpolation='none')
        figs.colorbar(im, ax=axes.ravel().tolist())
        file_name = plot_name + '_fmap' + str(j) + '.png'
        plt.savefig(os.path.join(file_path, file_name))
        plt.clf()
        plt.close()
    return


###################################################################################
class snn_layer():
    '''
    Dummy class from which snn_denseLayer and snn_convLayer inherit

    '''


###################################################################################

class snn_denseLayer(DenseLayer, snn_layer):
    def __init__(self, incoming, num_units, batch_size, stdp_enabled=True, threshold=64, refractory_voltage=-10000,
                 **kwargs):
        self.incoming = incoming
        self.batch_size = batch_size
        self.stdp_enabled = stdp_enabled
        self.threshold = threshold
        self.refractory_voltage = refractory_voltage
        super(snn_denseLayer, self).__init__(incoming, num_units, W=lasagne.init.Normal(std=0.01, mean=0.8), **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):
        self.input = input
        v = self.v_in + super(snn_denseLayer, self).get_output_for(input, **kwargs)
        vmax = T.max(v)
        flag = T.gt(vmax, self.threshold)
        self.output_spike = T.switch(T.eq(vmax, v), flag, 0.0)
        self.v_out = flag * self.refractory_voltage + (1.0 - flag) * v
        return self.output_spike

    def set_inputs(self, V, H_in):
        self.v_in = V
        self.H_in = H_in

    def get_output_shape(self):
        return (self.batch_size, self.num_units), lasagne.layers.get_output_shape(self.incoming), (
        self.batch_size, self.num_units)

    def do_stdp(self):
        self.H_out = self.H_in + self.input
        W = self.W  # Dx1024
        W = W.dimshuffle('x', 0, 1)  # 1XDX1024
        W = T.addbroadcast(W, 0)
        output_spike = self.output_spike
        output_spike = output_spike.dimshuffle(0, 'x', 1)
        output_spike = T.addbroadcast(output_spike, 1)
        H_out_reshaped = T.reshape(self.H_out, (T.shape(self.H_out)[0], -1, 1))
        H_out_reshaped = T.addbroadcast(H_out_reshaped, 2)

        update = ((W * (1 - W)) * output_spike * T.switch(T.eq(H_out_reshaped, 0.0), -1.0, 1.0))
        sum_update = T.sum(update, axis=0)
        count_nez = T.switch(T.eq(update, 0.0), 0.0, 1.0)

        count_update = T.sum(count_nez, axis=0)

        count_update = T.switch(T.eq(count_update, 0.0), 1, count_update)
        count_update = T.cast(count_update, dtype=theano.config.floatX)

        update = sum_update / count_update

        self.update = update

        return self.update, self.H_out


###################################################################################

class snn_convLayer(ConvLayer, snn_layer):  # 这里显式继承了object是新式类，不过snn_layer 是虚拟类
    def __init__(self, incoming, stdp_enabled=False, threshold=10,
                 num_filters=4, filter_size=5, output_flag=1, refractory_voltage=-np.float32(10000000), **kwargs):
        # incoming 是输入层信息，后面的params有五个元素分别代表之后的五个
        # super(DoG_Layer, self).__init__(incoming, **kwargs)
        self.v_in = T.fvector()
        # self.H_in = T.fvector()  # 参数在实例化的时候会传入，但是那样不太好，所以加在此处
        # self.v=v.astype(theano.config.floatX)
        # _,self.channels,self.height,self.width=incoming.shape
        self.incoming = incoming  # (1,2,28,28)?
        self.num_filters = num_filters
        # self.batch_size=incoming.input_var.shape[0]
        # self.batch_size=batch_size
        self.stdp_enabled = stdp_enabled
        self.output_flag = output_flag
        self.threshold = threshold
        self.refractory_voltage = refractory_voltage
        # self.output_spike=T.zeros([1,num_units])
        # w初始化是正态分布u=mean（均值），delta = std（标准差），same表示卷积后图像不降维,identity f(x)=x
        super(snn_convLayer, self).__init__(incoming, num_filters, filter_size, pad='same', flip_filters=False,
                                            W=lasagne.init.Normal(std=0.04, mean=0.8), b=lasagne.init.Constant(0.),
                                            nonlinearity=lasagne.nonlinearities.identity, **kwargs)

    def convolve(self, input, deterministic=False, **kwargs):
        # 重载Conv2DLayer中的convolve函数，用来计算脉冲卷积
        # print(super(snn_denseLayer, self).get_output_for(input, **kwargs))
        self.input = input  # 输入数据是归一化的，权重符合正态分布
        v = self.v_in + super(snn_convLayer, self).convolve(input, **kwargs)
        # 这里super使用了父类的convolve函数，就是普通的卷积，输出数据形状就是网络的最终输出形状，(1,30,28,28)
        self.v_in = v
        # vmax=theano.tensor.signal.pool.pool_3d(v, ds=(3,3,self.num_filters), ignore_border=True,
        #                                     st=(1,1,1), padding=(1, 1, 0), mode='max',)
        shape = T.shape(v)

        # 这里实现了降维，沿着第二个维度（最后降为1维）取最大值，成为(1,1,28,28)
        vmax, arg_max = T.max_and_argmax(v, axis=1, keepdims=True)
        self.arg_max = arg_max  # v最大对应该维度上的位置索引（单个数字，因为最大肯定会降到一维），总体shape与vmax一致
        # channelwise
        if self.stdp_enabled is False:

            tmp = T.switch(T.gt(vmax, self.threshold), 1.0, 0.0)  # (1,1,28,28) ,超阈值置1，
            output_spike = tmp * T.eq(T.arange(self.num_filters).dimshuffle('x', 0, 'x', 'x'), arg_max)
            # 广播点×，(1,30,28,28)，后面的eq相当于一个拉伸，每个filter数字对应位置(arg_max决定)为真，是为脉冲发射信号
            v2 = (1 - tmp) * v + tmp * self.refractory_voltage * (
                    1 - output_spike) + tmp * self.refractory_voltage * output_spike
            # filters之间同位置的互相抑制WTA，发射脉冲的那一像素道全都复位，其他保持原来电位，感觉这里output_spike更大的作用是扩维0.0
            self.output_spike = output_spike
            self.v_out = v2

            self.temp2 = v2
            self.tempy = v2
            self.tempx = v2

        else:
            print('stdp enabled')
            # 最大池化操作将在input的最后两个维度进行, pad决定边会扩充，所以不会降维,(1,1,28,28)
            temp2 = T.signal.pool.pool_2d(vmax, ws=(3, 3), ignore_border=True, stride=(1, 1), pad=(1, 1), mode='max')
            # B x 1 x H x W
            # print('***********'+str(temp2))
            # 这里通过T.eq操作扩维，再重组为(1,2,28*28)，平铺
            temp3 = T.reshape(T.switch(T.eq(temp2, v), v, 0.0), (shape[0], shape[1], -1))
            # 取第三维的最大值（分别是图片的两个通道的最大值）(1,2,1)， vs决定每个通道释放脉冲的具体是哪个N
            v_spatial, v_spatial_argmax = T.max_and_argmax(temp3, axis=2, keepdims=True)
            # 该通道超过阈值的最大值置1,thresh1决定通道是否有脉冲释放
            thresh1 = T.gt(v_spatial, self.threshold).astype(theano.config.floatX)
            # B x C x 1，temp2的图片维平铺，超出阈值的取1
            thresh2 = T.gt(T.reshape(temp2, (shape[0], 1, -1)), self.threshold).astype(theano.config.floatX)
            # B x 1 x HW，dimshuffle函数用于扩维，‘x’代表增加一维，0代表原张量的0维度，还可以有1,2,3等，还可以互换位置
            # 将v_s对应的索引还原
            output_spike = T.reshape(
                (T.eq(T.arange(shape[2] * shape[3]).dimshuffle('x', 'x', 0), v_spatial_argmax) * thresh1), shape)
            # 释放脉冲的位置为1
            flag = T.ge(thresh1 + thresh2, 1.0)  # B x C x HW
            # 释放了脉冲(唯一，可能有多个N达到阈值)的通道该通道N电位全体复原，没有释放过脉冲的通道，两通道对应的超出阈值的N电位复原(另一个通
            # 道已经全体复原，就不用额外操作了)，如果两通道都没有释放脉冲，这种情况也不会有N电位超出阈值，不用复原
            temp4 = T.reshape(T.switch(flag, self.refractory_voltage, T.reshape(v, (shape[0], shape[1], -1))), shape)

            temp3 = T.eq(T.arange(self.num_filters).dimshuffle('x', 0, 'x', 'x'), arg_max) * v

            self.temp2 = temp2
            self.tempy = thresh1
            self.tempx = temp4  # 处理后电位

            self.output_spike = output_spike
            self.v_out = temp4

        if self.output_flag == 1:

            return self.output_spike

        else:
            return self.v_out

    def do_stdp(self):
        self.H_out = self.H_in + self.input
        w_update = stdpOp()(self.output_spike, self.H_out, self.W)
        w_update = T.mean(w_update, axis=0)
        self.update = w_update

        return self.update, self.H_out

    def get_output_shape(self):  # 这里把本层输入incoming的形状(取决于前层)放在[1]，,本层输出放在[0]
        input_shape = lasagne.layers.get_output_shape(self.incoming)
        return (input_shape[0], self.num_filters, input_shape[2], input_shape[3]), input_shape


####################################################################################

class snn_poolLayer(PoolLayer, snn_layer):
    def __init__(self, incoming, stride=6, filter_size=5, refractory_voltage=-np.float32(10000000), threshold=0.99,
                 **kwargs):
        self.incoming = incoming
        self.stdp_enabled = False
        self.threshold = threshold
        self.refractory_voltage = refractory_voltage
        self.v_in = T.fvector()
        # self.output_spike=T.zeros([1,num_units])
        super(snn_poolLayer, self).__init__(incoming, filter_size, stride)

    def get_output_for(self, input, deterministic=False, **kwargs):
        # print(super(snn_denseLayer, self).get_output_for(input, **kwargs))
        self.input = input
        v = self.v_in + super(snn_poolLayer, self).get_output_for(input, **kwargs)
        self.v_in = v
        # vmax=theano.tensor.signal.pool.pool_3d(v, ds=(3,3,self.num_filters), ignore_border=True,
        #                                     st=(1,1,1), padding=(1, 1, 0), mode='max',
        #                                       )

        # channelwise

        self.output_spike = T.ge(v, self.threshold) * np.float32(1.0)
        self.v_out = T.switch(self.output_spike, self.refractory_voltage * np.float32(1.0), v * np.float32(1.0))

        return self.output_spike

    def do_stdp(self):
        self.H_out = self.H_in + self.input
        w_update = stdpOp()(self.output_spike, self.H_out, self.W)
        w_update = T.mean(w_update, axis=0)
        self.update = w_update

        return self.update, self.H_out

    def get_output_shape(self):
        input_shape = lasagne.layers.get_output_shape(self.incoming)
        output_shape = lasagne.layers.get_output_shape(self)
        return output_shape, input_shape


###################################################################################


###################################################################################

class snn():
    def __init__(self, input_shape, input=None, num_class=10, name='snn_autonet'):
        if input is None:
            input = T.tensor4()  # 占位，int32，dim=4，(?,?,?,?)
        self.input_shape = input_shape
        self.input = input
        self.name = name
        self.num_class = num_class
        self.time_steps = 32
        self.batch_size = self.input_shape[0]   # 第一个参数就是批量的大小

        print 'computing DoG maps ...'
        # 定义了关于dog的函数
        self.DoG_maps = self.dog_output(input)
        # the input passed to the class is simply the image, the DoG maps are calculated using
        # this image. A slice of this DoG map  is given as the input to the graph created
        # 返回一个TensorVariable，用时间编码好的一系列（与time_steps一致）脉冲发放的张量(32,1,2,28,28)，0/1值
        self.get_dog_map = theano.function(inputs=[input], outputs=self.DoG_maps)
        print 'Done!'

        self.input_shape = list(self.input_shape)  # 因为元组不能更改，所以先转换成列表
        self.input_shape[1] = self.input_shape[1] * 2  # 变成了两通道???

        self.input_shape = tuple(self.input_shape)  # (1,2,28,28)

        # lasagne包的Input函数, This layer holds a symbolic variable that represents a network input. compile time.
        input_layer = InputLayer(shape=self.input_shape, input_var=T.reshape(self.DoG_maps[0], self.input_shape))
        # InputLayer只用于接收数据，不对数据做任何处理，起到占位符的作用。多维张量索引DoG[0]维度变为(1,2,28,28)
        # input_var表示需要连接到网络输入层的theano变量，就是它在占位。 这里就是实例化操作。
        self.layers = [
            ('snn_conv1', snn_convLayer, [False, 15, 30, 5, 1]),
            # 对应卷积层的输入参数为stdp_enabled,threshold, num_filter,filter_size,output_flag
            ('pool1', snn_poolLayer, [2, 2]),  # stdp_enabled,stride,filter_size
            ('snn_conv2', snn_convLayer, [False, 20, 100, 5, 1])
            # ('gap_layer',GapLayer,[T.max])
        ]
        # 不包括输入层的其它层
        layer_head = self.layers
        layers = self.layers
        self.layer_names = ['input'] + [attr[0] for attr in layers]  # 按层排列的名字列表

        full_net_layers, _ = self.create_net(layers, input_layer)  # 进行带参数实例化，一个列表，排好的实例化的各层
        # print 'loading snn'
        # model_save_path = './models'
        # path = os.path.join(model_save_path, 'train1')
        # print path
        # f = open(os.path.join(path, 'snn_autonet' + '.save'), 'rb')
        # snn_loaded_object = cPickle.load(f)
        # f.close()
        # print('Done')

        # self.copy_nets(snn_loaded_object.full_net_layers[-2], full_net_layers[-2])
        self.full_net_layers = full_net_layers

        self.small_networks = []
        # initial_train,initial_train=create_net(layers) # initial_train not to be used
        snn_net_train_funcs = []
        snn_net_test_funcs = []
        weight_list = []

        snn_net_split_on_layer_list = [snn_convLayer, snn_denseLayer]

        j = 0

        if layers[0][1] not in snn_net_split_on_layer_list:   # [0][1]是类实例snn_conv1
            i = 1
        else:
            i = 0
        for i in range(i, len(layer_head) + 1):   # 对snn_conv层和snn_dense层进行如下操作，其余层会跳出当次循环
            if i < len(layer_head) and layers[i][1] in snn_net_split_on_layer_list:  # 这里看似是跳出有conv层的循环，
                # 但恰好其他正常循环时，得到的tmp_layers的[-1]是conv层
                continue
            tmp_layers = layers[: i]  # + layer_tail，在这里程序只会得到，i=1,和i=3的值，layers还未进行实例化
            tmp_layers[-1][2][0] = True    # 这样[-1]分别是两个conv层，使能STDP
            # print 'layers ',layers
            print 'tmp_layers:', tmp_layers
            print 'Creating: ', [l[0] for l in tmp_layers]  # 每一层的[0]是该层的名字
            lyr_list, _ = self.create_net(tmp_layers, input_layer)  # 比较像全连接层的创建，但这里的conv层是使能STDP的
            self.small_networks.append(lyr_list)
            self.copy_nets(full_net_layers[-1], lyr_list[-1])  # 复制full所有对应层参数到lyr,将使能STDP的conv层参数params（W,b）
            # 替换为之前创建的全连接层的， 而输入层INPUT没有params，所以虽然也完成了层复制，但实际上没有复制任何东西。
            # 虽然只给了-1，但copy内会ask所有前层

            '''for k in lyr_list:
            try :
            print k.W.get_value()
            print 'in small network '+str(j)
            except:
            print '''

            train, test, _ = self.create_snn(lyr_list)   # 建立脉冲网络,lyr_list带输入层
            snn_net_train_funcs.append(train)
            snn_net_test_funcs.append(test)
            weight_list.append(lyr_list[-1].W)  # 这里只取了最后一层的权值
            tmp_layers[-1][2][0] = False  # 否定最后一层的STDP，感觉没啥后续意义？
            j = j + 1
            print '*****************************' + 'small_network ' + str(
                j) + ' created' + '****************************'

        i = len(layer_head)
        tmp_layers = layers[: i]  # + layer_tail
        tmp_layers[-1][2][1] = np.float32(1000000000)  # set threshold to infinity
        tmp_layers[-1][2][-1] = 0  # set output flag to 0 to get voltage

        print 'tmp_layers:', tmp_layers
        print 'Creating: ', [l[0] for l in tmp_layers]
        lyr_list, _ = self.create_net(tmp_layers, input_layer)
        self.small_networks.append(lyr_list)
        self.copy_nets(full_net_layers[-1], lyr_list[-1])

        _, _, self.get_final_voltage = self.create_snn(lyr_list)

        self.train_funcs = snn_net_train_funcs
        self.test_funcs = snn_net_test_funcs
        self.weight_list = weight_list

        print 'train_funcs', self.train_funcs
        print 'test_funcs', self.test_funcs

        # self.create_snn()

    def dog_output(self, input_image):

        _, self.channels, self.height, self.width = input_image.shape

        # 卷积,flip代表是否翻转，翻转是卷积操作，subsample是步长，dog_W给出的是fliter的规格(2,1,7,7), 实际上这里是两个滤波器
        # 滤波器参数意义(output channels, input channels, filter rows, filter columns)
        # 返回 Set of feature maps generated by convolutional layer.Tensor is of shape(batch size, output channels,
        # output rows, output columns)
        conv_output = T.nnet.conv2d(input_image, dog_W, filter_flip=False,
                                    border_mode='half', subsample=(1, 1))

        # 经过卷积层处理，这里输出通道与该卷积层的filter的数目一致，即output channels，2
        # 此处将通道按行反向排列，在第二维处做更改，后面两维作为整体不变，第一维也不受影响。
        conv_output2 = conv_output[:, ::-1, :, :]
        # 高斯差分滤波器是用来对灰度图增强和角点的方法，一般用在边缘检测，或者图片的分割预处理
        dog_maps = conv_output - conv_output2  # difference of guassians (1,2,28,28)
        # 这里其实是两幅DoG图，positive和negative，就是差个负号

        dog_maps = T.ge(dog_maps, 0) * dog_maps   # ge是大于等于的意思
        dog_maps = T.switch(T.ge(dog_maps, T.mean(dog_maps)), dog_maps, 0.0)
        # theano.tensor.switch(cond, ift, iff)
        # 最后的DoG图是一个大于等于正数均值的数组，非逻辑值

        # dog_maps=T.set_subtensor(dog_maps[:,1:2,:,:],np.float32(0.0))
        sorted_dog = T.sort(T.reshape(dog_maps, (-1,)))
        # -1默认为一行, sort默认对每行由低到高排序输出一个array，实在不明白为什么这里排成了一个向量，不会丢失位置信息么
        # 已了解，这里排序只是为了后面确定bin_limits分区的上下限，不会用来判断脉冲的发放时间
        # sorted_dog=T.shape(sorted_dog)-T.sum(T.neq(sorted_dog,0.0))
        num_spikes = T.neq(sorted_dog, 0.0)  # 不等于0时释放脉冲,输出一个逻辑行向量
        num_spikes_per_bin = T.sum(num_spikes) // self.time_steps  # 单位时间的发放脉冲数，这里把脉冲发放总数均匀映射到整个时间域
        # 即每一时间步长里发放相同的脉冲数，只是发放的神经元不一样
        i = T.shape(sorted_dog)[0] - num_spikes_per_bin  # 这是一张图片的DoG变成了一行，所以这里的0给出的是像素数
        bin_limits = T.zeros(self.time_steps + 1)   # 一行33的0值向量
        bin_limits = T.set_subtensor(bin_limits[0], sorted_dog[-1])  # 给张量bin..填充sort..的值
        for j in range(0, self.time_steps):
            bin_limits = T.set_subtensor(bin_limits[j + 1], sorted_dog[i])  # 感觉像是每隔单位时间脉冲发放数采样一次sort_dog
            # 这里应该是保证时间步长不变，把排序好的sd图相应时间上的数值复制到bl里，作为分区的上下限
            i = i - num_spikes_per_bin

        # return dog_maps,sorted_dog,bin_limits
        # return self.temporal_encoding(dog_maps,bin_limits)
        return T.reshape(self.temporal_encoding(dog_maps, bin_limits) * np.float32(1.0),
                         [self.time_steps, self.batch_size, self.channels * 2, self.height, self.width])

    # return output

    def temporal_encoding(self, dog_maps, bin_limits):
        def fn(*args):
            print('args')
            print(args)  # args[0]=bin_limits[0],args[1]=bin_limits[1],即给出一个上下限
            output = T.le(dog_maps, args[0]) * np.float32(1.0) * T.gt(dog_maps, args[1])   # le小于等于，gt大于
            # 落入该分区的DoG图的数值会输出1
            return output

        temporal_encoding, _ = theano.scan(fn, sequences=[dict(input=bin_limits, taps=[0, 1])], non_sequences=dog_maps,
                                           outputs_info=T.zeros_like(dog_maps, dtype=theano.config.floatX))
        # do_encoding = theano.function(inputs=[dog_maps,bin_limits],outputs=temporal_encoding)

        # #print('compiled')
        # #output = do_encoding(dog_maps.eval(),bin_limits.eval())
        return temporal_encoding  # 将是一个按时间排列的各像素点的脉冲发放情况

    def plot_weights(self, stage_id, file_path='.', plot_id=0,
                     max_subplots=64, max_figures=64, layer_id=-1,
                     figsize=(6, 6)):
        i = -1
        W = self.weight_list[stage_id]
        # W = T.reshape(W,(1,2,28,28))
        plot_weights(W, 'ID' + str(plot_id) + '_' + 'snn', file_path, max_subplots, max_figures, figsize)

    def copy_nets(self, net1, net2, net1_inputs=[], net2_inputs=[]):
        print 'INSIDE: copy_nets'

        def replace_var(obj, var, new_var):
            replaced = False
            for key, value in obj.__dict__.iteritems():  # 应该是对象的命名空间里，将特定的var替换为new_var
                if value == var:
                    obj.__dict__[key] = new_var
                    replaced = True
            return replaced

        net1_layers = lasagne.layers.get_all_layers(net1, net1_inputs)  # 就是得到这层之前的所有依赖层，也包括这层
        net2_layers = lasagne.layers.get_all_layers(net2, net2_inputs)
        # print net2_layers

        lyr_idx = 0
        num_layers_copied = 0
        num_vars_copied = 0
        for lyr_dest in net2_layers:
            if lyr_dest in net2_inputs:
                continue
            if type(net1_layers[lyr_idx]) == type(lyr_dest):
                p_dest_list = lyr_dest.params
                p_src_list = net1_layers[lyr_idx].params
                for p_dest, p_src in zip(p_dest_list, p_src_list):  # 确定参数个数以及参数shape一致
                    p_dest = p_dest.get_value()
                    if isinstance(p_dest, np.ndarray):
                        print p_dest.shape, p_src.get_value().shape
                        assert (p_dest.shape == p_src.get_value().shape)
                    else:
                        assert 0

                assert (len(p_dest_list) == len(p_src_list))
                lyr_dest.params = p_src_list
                for p_src, p_dest in zip(p_src_list, p_dest_list):  # 开始替代参数
                    flag = replace_var(lyr_dest, p_dest, p_src)  # 将目标层的目标参数替换为源参数，返回逻辑值
                    assert flag
                    num_vars_copied += 1
                num_layers_copied += 1
                lyr_idx += 1
        print 'Number of variables copied:', num_vars_copied
        print 'Number of layers copied:', num_layers_copied
        print lyr_idx, len(net1_layers), len(net2_layers)
        if lyr_idx != len(net1_layers):
            print 'WARNING: NOT ALL LAYERS FROM net1 COPIED TO net2.'
        # assert(lyr_idx == len(net1_layers))
        return True

    def create_net(self, layer_details, prev_layer):
        print self.name, '.create_net> building net...'
        layers_list = [prev_layer]
        layers_dict = {'input': prev_layer}
        for attributes in layer_details:
            print 'attributes : ', attributes
            name, layer_fn = attributes[: 2]  # 切片操作这里不包括2，[1]是一个layer类，在这里相当于起了个别名layer_fn
            params = []
            params_dict = {}
            if len(attributes) >= 4:
                params, params_dict = attributes[2: 4]
            elif len(attributes) >= 3:
                params = attributes[2]       # 设置实例化每层时的几个参数，[stdp使能，阈值，滤波器数，大小，输出标志]
            print 'layer: ', name
            prev_layer = layer_fn(prev_layer, *params, **params_dict)
            # 实例化layer_fn, 循环的过程中，prev_layer 一直在更新， 作为下一层的incoming
            layers_dict[name] = prev_layer
            layers_list.append(prev_layer)
        print 'done!'
        return layers_list, layers_dict

    def save_model(self, file_name=None, layer_to_save=None):
        if layer_to_save is None:
            layer_to_save = self.layer_list[-1]
        # assert(layer_to_save == self.net['bu_softmax'])
        print 'Saving model starting from layer', layer_to_save.name, '...'
        print 'filename', file_name
        params = lasagne.layers.get_all_param_values(layer_to_save)
        if file_name is not None:
            fp = open(file_name + '.save', 'wb')
            cPickle.dump(params, fp, protocol=cPickle.HIGHEST_PROTOCOL)
            fp.close()
        print 'Done.'
        return params

    def load_model(self, file_name, layer_to_load=None):
        if layer_to_load is None:
            layer_to_load = self.layer_list[-1]
        print 'Loading model starting from layer', layer_to_load.name, '...'
        fp = open(file_name, 'rb')
        params = cPickle.load(fp)
        fp.close()
        lasagne.layers.set_all_param_values(layer_to_load, params)
        print 'Done'

    def create_snn(self, layers):
        print 'Building snn...'
        # if(layers=='None'):
        #     layers=self.layers
        # input_layer = InputLayer(shape = self.input_shape, input_var =  T.reshape(self.DoG_maps[0],self.input_shape))
        # the input layer of
        # #the graph which takes a slice of DoG map.
        # all_layers, _ = self.create_net(layers, input_layer)
        all_layers = layers  # 该layers带输入层
        # self.all_layers=all_layers
        lr = T.scalar()

        def fn(*args):  # 一旦被创建就不会再进入这里了，所以不能看出迭代次数

            args = list(args)
            print args
            print(len(args))
            print('args')
            print(args)
            i = 2
            for layer in (all_layers[1:]):  # 除了INPUT层，给出这些层参数的初始值
                if isinstance(layer, snn_layer):  # 是虚拟层snn的实例，conv和pool都是

                    if layer.stdp_enabled:
                        layer.v_in = args[i]  # conv层定义的属性,这里给网络输入初始参数
                        layer.H_in = args[i + 1]  # 在类外定义实例属性H_in
                        i += 2
                    else:  # 作为test函数的返回值
                        layer.v_in = args[i]
                        i += 1
                # 对于两层网络lyr_list，conv1：vin=args[2],hin=args[3]
                # 对于四层网络lyr_list，conv1同上，pool1:vin=args[4],conv2:vin=args[5],hin=args[6],
            all_layers[0].input_var = args[0]  # arg[0]应该是输入的DoG图[0]，给出整个网络的输入信息，下面才能使用get_output
            # #all_layers[0].input_var=T.reshape(args[0],(1,2,28,28))
            output_spike_train = lasagne.layers.get_output(all_layers[-1])  # the graph is created
            # 会作为之后迭代的args[1]，得到卷积网络的输出，中间应该调用了卷积层的convolve，下面基于这个输出进行STDP学习
            # Computes the output of the network at one or more given layers.
            # 这个helper fn返回的是convolve的值，与output flag有关，=0时，返回电压
            # theano.printing.pydotprint(output_spike_train, outfile="./spike_graph.png", var_with_name_simple=True)


            # STDP
            vH_out_list = []
            # H_out_list=[]
            W_dict = []

            for layer in all_layers[1:]:
                if isinstance(layer, snn_layer):
                    vH_out_list.append(layer.v_out)
                    if layer.stdp_enabled:
                        layer.do_stdp()  # 层内定义的方法，返回self.update, self.H_out
                        vH_out_list.append(layer.H_out)  # vH_out的更新规则
                        W_dict.append((layer.W, layer.W + lr * layer.update))   # 权值的更新规则, (30,2,5,5)
            print('fn returning : ')
            # for k in [output_spike_train] + vH_out_list:
            #     print(k)
            output = [output_spike_train] + vH_out_list
            return output, W_dict
            # 函数需要返回两个东西。一个是按照 outputs_info 顺序排列的输出列表，不同的是对每个输出初始状态只有一个输出变量对应
            # （即使没有使用 tap 值）。fn还应当返回一个更新字典（告诉程序如何对共享变量进行每步的更新），字典也可以以 tuple 的列表给出。


        def set_outputs_info():  # 确定输出的初始状态，之后的scan不会再用到
            output = []

            # initial_spike_train=T.zeros(all_layers[-1].get_output_shape()[2])
            # initial_spike_train=T.zeros((self.batch_size,32,28,28))
            initial_spike_train = T.zeros(lasagne.layers.get_output_shape(all_layers[-1]))
            # 得到输出层形状张量，-1是conv1层(1,30,28,28)
            # initial_spike_train=T.zeros((self.batch_size,self.all_layers[-1].num_units))

            print T.shape(initial_spike_train)

            # output.append(initial_spike_train)

            vH_list = []

            # for layer in all_layers[1:]:
            #     layer.set_inputs(T.vector(),T.tensor4())

            for layer in all_layers[1:]:
                if isinstance(layer, snn_layer):
                    # print(T.zeros(layer.get_output_shape()[0])
                    vH_list.append(T.zeros(layer.get_output_shape()[0]))  # shape[0]得到（1,30,28,28）
                    if layer.stdp_enabled:
                        print 'use stdp'
                        vH_list.append(T.zeros(layer.get_output_shape()[1]))  # shape[1]得到[1,2,28,28]，返回该层的输入shape

            output = [initial_spike_train] + vH_list
            # output=vH_list

            # print(output)
            # output = [T.shape_padleft(a) for a in output]
            # for i,a in enumerate(output):
            #     output[i]=T.shape_padleft(a)

            print('set output info :')
            print(output)
            # print(T.shape(output))
            return output


        # theano.printing.pydotprint(self.DoG_maps, outfile="./debug.png", var_with_name_simple=True)
        components, updates = theano.scan(fn, sequences=[self.DoG_maps], non_sequences=lr,
                                          outputs_info=set_outputs_info())
        # 带入fn参数的顺序为 sequences, outputs_info, non_sequences。fn可以返回sequences变量的更新updates
        # DoG图是与time_step长度一致的列表，每个元素是 该时间内应该释放脉冲的神经元标记1
        # components 包括了所有迭代步骤的输出结果, [-1]是最后的结果。结构为[output_spike_train] + vH_out_list，
        # 这里有三项内容，每项中包含了自己的所有迭代结果，升维
        # updates 是一个字典的子类指定了所有共享变量的更新规则。这个字典应该被传递给 theano.function。
        # 不同于正常的字典的是我们验证这些 key 为 SharedVariable 并且确保这些字典的求和是一致的。
        # updates的参数需要一个list参数列表对，形式是这样的[(key, value), (key, value), .......]
        # 这样的一个列表但是key必须是shared_variable也就是共享变量，value是一个新的表达式

        # print(T.shape(components))
        shape = T.shape(components[0])
        # 这里的components[0]不一定是spike，也可能是电压
        output = T.sum(components[0], axis=0)  # [0]是输出脉冲的所有时间片序列,(32,1,30,28,28)
        # len(components)=3，输出有三项内容，每项ndim=5
        output = T.switch(T.ge(output, 1.0), 1.0, output)  # 到底是谁释放了脉冲？
        output = T.cast(output, dtype=theano.config.floatX)  # 128x1024 ，类型转换

        delta_weight = T.zeros(1)
        # print('*********')
        # print delta_weight
        i = 0
        for key, value in updates.iteritems():  # iteritems同时迭代键W(30,2,5,5)和键值
            i += 1
            delta_weight += T.mean(abs(value - key))  # 好像在构建损失函数，dW是一个一维张量。这里只迭代了一次，将整个W同时求mean

        delta_weight /= len(updates.keys())  # 四层网络时会有两本字典，分别对应两个conv层的权值更新

        get_voltage = theano.function(inputs=[self.input, lr], outputs=components[0][-1], updates=updates)
        # to be used only when output flag is 0，只有是0的时候才会返回voltage

        train = theano.function(inputs=[self.input, lr], outputs=[components[0], delta_weight], updates=updates,
                                on_unused_input='ignore')

        if layers[-1].stdp_enabled == False:
            test = theano.function(inputs=[self.input, lr], outputs=output)
        else:
            print('recursive call')
            layers[-1].stdp_enabled = False
            _, test, _ = self.create_snn(layers)

        print('compiled')

        return train, test, get_voltage

    def test_batch(self, X):
        return self.test_funcs[1](X.astype(theano.config.floatX), 0.0).astype(theano.config.floatX)
        # self.get_weights=theano.function([],all_layers[-1].W)

        # def test_batch(self,X):
        #     classifier_input=np.zeros((np.shape(X)[0],1024),dtype=theano.config.floatX)
        #     for i in range(0,np.shape(X)[0]):
        #         slice_output=self.test(X[i:i+1],0)
        #         classifier_input[i]=slice_output
        #
        #     return classifier_input


####################################################################################


if __name__ == '__main__':
    np.random.seed(64)

    batch_size = 1
    data_path = 'mnistdata/'

    print 'Loading Dataset'
    mnist = mnist_data_set(data_path, batch_size)
    print 'Done!'

    datasets = mnist.data_sets
    print type(datasets['train'].X)  # X代表图片，Y代表标签
    print (datasets['train'].X)
    print np.max(datasets['train'].X), np.min(datasets['train'].X)

    data_shape = datasets['train'].X.shape
    # print type(data_shape) #shape得到元组类型
    data_shape = (batch_size,) + data_shape[1:]  # 数据shape的第一个参数，即[0]，是输入大小，
    # 这里通过元组的加法换成批量数，改变数据形状
    print 'Data shape:', data_shape

    # k = T.iscalar('k')
    # A = T.vector('A')
    #
    # outputs, updates = theano.scan(lambda result, A: result * A,
    #                                non_sequences=A, outputs_info=T.ones_like(A), n_steps=k)
    #
    # fn_Ak = theano.function([A, k], outputs, updates=updates)
    # print fn_Ak(range(10), 4)

    network = snn(data_shape)

    # snn_object=snn(data_shape)
