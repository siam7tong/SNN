# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import timeit
# import pickle
import cPickle
import os
import datetime
# import cv2
import lasagne
import random
import matplotlib
from numpy import dtype
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from theano.compile.nanguardmode import NanGuardMode


from mnist_reader import data_set, mnist_data_set


from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import GlobalPoolLayer as GapLayer
from lasagne.nonlinearities import softmax, sigmoid
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import batch_norm
from lasagne.nonlinearities import rectify
import scipy.io as sio

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from stdp import stdpOp
from itertools import product as iter_product

on_unused_input = 'ignore'

import sys


# from visualize import plot_conv_weights,

def relu1(x):
    return T.switch(x < 0, 0, x)


def std_conv_layer(input, num_filters, filter_shape, pad='same', nonlinearity = lasagne.nonlinearities.rectify,
                   W=None,
                   # W = lasagne.init.Normal(std = 0.01, mean = 0.0),
                   b=lasagne.init.Constant(0.),
                   do_batch_norm=False):
    if W is None:
        if nonlinearity == lasagne.nonlinearities.rectify:
            print 'convlayer: rectifier func'
            W = lasagne.init.HeNormal(gain='relu')
        else:
            print 'convlayer: sigmoid func'
            W = lasagne.init.HeNormal(1.0)
    else:
        print 'convlayer: W not None'
    conv_layer = ConvLayer(input, num_filters, filter_shape, pad=pad, flip_filters=False, W=W, b=b,
                           nonlinearity=nonlinearity)
    if do_batch_norm:
        conv_layer = lasagne.layers.batch_norm(conv_layer)
    else:
        print 'convlayer: No batch norm.'
    return conv_layer

from snn_conv import snn
from lasagne import layers

from lasagne.layers import TransposedConv2DLayer as DeconvLayer
from lasagne.layers import ExpressionLayer
# try:
#
# except:
#     from new_conv import TransposedConv2DLayer as DeconvLayer
#
#
# try:
#
# except:
#     from new_special import ExpressionLayer


###############################################################################

W1 = np.array([[1.96519161e-05, 2.39409349e-04, 1.07295826e-03, 1.76900911e-03,
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

W2 = np.array([[0.00492233, 0.00919613, 0.01338028, 0.01516185, 0.01338028, 0.00919613, 0.00492233],
               [0.00919613, 0.01718062, 0.02499766, 0.02832606, 0.02499766, 0.01718062, 0.00919613],
               [0.01338028, 0.02499766, 0.03637138, 0.04121417, 0.03637138, 0.02499766, 0.01338028],
               [0.01516185, 0.02832606, 0.04121417, 0.04670178, 0.04121417, 0.02832606, 0.01516185],
               [0.01338028, 0.02499766, 0.03637138, 0.04121417, 0.03637138, 0.02499766, 0.01338028],
               [0.00919613, 0.01718062, 0.02499766, 0.02832606, 0.02499766, 0.01718062, 0.00919613],
               [0.00492233, 0.00919613, 0.01338028, 0.01516185, 0.01338028, 0.00919613, 0.00492233]])
W = np.stack((W1, W2), axis=0)
dog_W = np.reshape(W, (2, 1, 7, 7))

dog_W = dog_W.astype(theano.config.floatX)

################################################################################


def log(f, txt, do_print=1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')


#################################################################################

input = T.tensor4()

gap_network = lasagne.layers.InputLayer(shape=(1, 10, 40, 32), input_var=input)
gap_network = lasagne.layers.GlobalPoolLayer(gap_network, T.max)  # This layer pools globally across all
# trailing dimensions beyond the 2nd.
gap_output = lasagne.layers.get_output(gap_network)   # 后面的两维没有了，(1,10)

get_gap = theano.function(inputs=[input], outputs=gap_output)

print('GAP COMPILED')
################################################################################


class softmax_classifier:
    def __init__(self, name='softmax_classifer'):
        # self.snn_network=snn_network
        self.name = name
        # self.input_shape=input_shape
        self.build_classifer()


    def build_classifer(self):
        print('Building classifier...')
        self.input = T.matrix('inputs')  # 这里的输入应该是globalpool层的输出(1,100)


        target = T.ivector('targets')
        LR = T.scalar('LR', dtype=theano.config.floatX)

        input_layer = lasagne.layers.InputLayer(shape=(None, 100), input_var=self.input)

        # dense_layer=lasagne.layers.DenseLayer(input_layer,num_units=128,
        #                                 W=lasagne.init.HeNormal(1.0),
        #                                 nonlinearity=lasagne.nonlinearities.rectify)

        self.output_layer = lasagne.layers.DenseLayer(input_layer, num_units=10,
                                                      W=lasagne.init.Normal(std=0.01, mean=0.8),
                                                      nonlinearity=lasagne.nonlinearities.softmax)
        'units代表神经元个数，也就是输出数字是10个。LR分类器。维度(1,10)'

        print('Done!')

        print('Building theano functions...')
        self.params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
        # Returns a list of Theano shared variables or expressions that parameterize the layer.

        self.train_output = lasagne.layers.get_output(self.output_layer)
        # Computes the output of the network at one or more given layers.

        self.test_output = self.train_output  # 训练输出与测试输出一致
        Y = T.ivector()
        train_error = lasagne.objectives.categorical_crossentropy(self.train_output, Y).mean()
        '# mean()将一维张量降为零维张量，也就是标量'
        train_loss = train_error  # 感觉这里将Y中的数字转化成了与输出同维度，相应第Y个位置置1的向量
        train_accuracy = T.mean(T.eq(T.argmax(self.train_output, axis=1), Y), dtype=theano.config.floatX)
        # 精度很好理解，最大概率索引值与目标值一致，则置1，默认在0维度实现mean，即计算准确率
        test_loss = lasagne.objectives.categorical_crossentropy(self.test_output, Y).mean()
        test_accuracy = T.mean(T.eq(T.argmax(self.test_output, axis=1), Y), dtype=theano.config.floatX)
        # 测试的loss和accuracy完全照搬训练的定义

        self.test_func = theano.function(inputs=[self.input, Y], outputs=[test_loss, test_accuracy, self.test_output])

        LR = T.scalar()

        pre_updates = lasagne.updates.adam(train_loss, self.params, learning_rate=LR)
        momentum = 0.9
        post_updates = lasagne.updates.momentum(train_loss, self.params, learning_rate=LR, momentum=momentum)
        # Stochastic Gradient Descent(SGD) updates with momentum
        # velocity := momentum * velocity - learning_rate * gradient
        # param := param + velocity
        self.pre_train_func = theano.function(inputs=[self.input, Y, LR], outputs=[train_loss, train_accuracy],
                                          updates=pre_updates)
        self.post_train_func = theano.function(inputs=[self.input, Y, LR], outputs=[train_loss, train_accuracy],
                                              updates=post_updates)
        print 'Done!'

    def test(self, X, Y):
        loss, accuracy, confidences = self.test_func(X, Y)
        # print(np.shape(self.snn_network.test_batch(X)))
        return confidences, loss, accuracy

    def train(self, X, Y, LR, ar_flag):
        if ar_flag == 1:
            # k = 1
            # if k == 1:
            #     print 'using pre_train_func: adam'
            loss, accuracy = self.pre_train_func(X, Y, LR)
        else:
            # k = 2
            # if k == 2:
            #     print 'using post_train_func: momentum'
            loss, accuracy = self.post_train_func(X, Y, LR)
        return loss, accuracy
        # print(self.snn_network.test_batch(X))
#####################################################################################


def train_classifer(network, datasets, log_path, snn_loaded_object):
    snapshot_path = os.path.join(log_path, 'snapshots')

    LR = 0.00001
    f = open(os.path.join(log_path, 'train.log'), 'w')
    log(f, 'softmax classifer')
    log(f, 'Learning rates LR: %f ' % LR)
    # LR_list = [1.0, 0.1, 0.1, 0.01, 0.001, 0.0001, 0.0001, 0.0001]
    LR_list = [0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.00001, 0.00001]

    num_epochs = 6
    losses = np.zeros((2, num_epochs + 1))
    accuracies = np.zeros((2, num_epochs + 1))

    prev_loss = 100

    flag = 0  # 1 means adam
    ii = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        for i, (X, Y) in enumerate(datasets['train']):
            ii += 1

            # network.train(X, Y, LR)
            # 0好像代表的是学习率，这里前几层网络已经训练完毕，也就不需要学习了
            # X(1,1,28,28) ,Y(1,), slo(1,100,14,14), get_gap(1,100)
            # if flag == 1:
            #     _, _, _ = network.test(get_gap(snn_loaded_object.get_final_voltage(X, 0.0)), Y)
            #     if ii == 1:
            #         print 'i am in SGD'
            #         if flag == 1:
            #             momentum = 0.9
            #             network.params = lasagne.layers.get_all_params(network.output_layer, trainable=True)
            #             network_train_output = lasagne.layers.get_output(network.output_layer)
            #             y = T.ivector()
            #             network_train_loss = lasagne.objectives.categorical_crossentropy(network_train_output,
            #                                                                               y).mean()
            #             network_train_accuracy = T.mean(T.eq(T.argmax(network_train_output, axis=1), y),
            #                                             dtype=theano.config.floatX)
            #             lr = T.scalar()
            #             updates = lasagne.updates.momentum(network_train_loss, network.params, learning_rate=lr,
            #                                                momentum=momentum)
            #             network_train_func = theano.function(inputs=[y, lr],
            #                                                  outputs=[network_train_loss, network_train_accuracy],
            #                                                  updates=updates)
            #
            #     loss, accuracy = network_train_func(Y, LR)
            # else:
            #     if ii == 1:
            #         print 'i am in adadelta'
            loss, accuracy = network.train(get_gap(snn_loaded_object.get_final_voltage(X, 0.0)), Y, LR, flag)
            # input0 = get_gap(snn_loaded_object.get_final_voltage(X, 0.0))
            # input_layer = lasagne.layers.InputLayer(shape=(None, 100), input_var=input0)
            # output_layer = lasagne.layers.DenseLayer(input_layer, num_units=10,
            #                                          W=lasagne.init.Normal(std=0.01, mean=0.8),
            #                                          nonlinearity=lasagne.nonlinearities.softmax)
            #
            # train_output = lasagne.layers.get_output(output_layer)
            #
            # print train_output.eval()
            # print Y
            # print 'loss', loss

            train_loss += loss

            train_accuracy += accuracy
            if ii % 2000 == 0:
                log(f, 'Iter: %d [%d], loss: %f, acc: %.2f%%, ' 'avg_loss: %f, avg_acc: %.2f%%'
                    % (ii, epoch, loss, accuracy, train_loss / (i + 1), 100.0 * (train_accuracy / (i + 1))))

        train_loss /= i
        train_accuracy /= i
        losses[0, epoch] = train_loss
        accuracies[0, epoch] = train_accuracy

        log(f, '\nEpoch %d: avg_Loss: %.12f, avg_Acc: %.12f'
            % (epoch, train_loss, train_accuracy * 100.0))

        epoch += 1

        # log(f, 'Learning rates changed LR: %f ' % LR)
        try:
            LR = LR_list[epoch]
            log(f, 'Learning rates changed LR: %f ' % LR)
        except:
            print 'Something is wrong'

        prev_loss = train_loss

        if train_loss == 0.0:
            break

        p1, = plt.plot(losses[0, : epoch], label='Training loss')
    
        plt.legend()
        plt.ylabel('LOSS')
        plt.xlabel('EPOCH NUMBER')
        plt.savefig(os.path.join(snapshot_path, 'losses.eps'))
        plt.clf()
        plt.close()

        p1, = plt.plot(100*accuracies[0, : epoch], label='Training accuracy')

        plt.legend()
        plt.ylabel('ACCURACY %')
        plt.xlabel('EPOCH NUMBER')
        plt.savefig(os.path.join(snapshot_path, 'accuracy.eps'))
        plt.clf()
        plt.close()

    np.save(os.path.join(snapshot_path, 'losses.npy'), losses)
    '''
    min_epoch = np.argmin(losses)
    log(f, 'Done Training.\n Minimum loss %f at epoch %d' %
        (losses[min_epoch], min_epoch))
    # '''
    # log(f, '\nTesting at last epoch...')
    # _, _, txt = test_classifier(network, datasets['test'], snn_loaded_object)
    # log(f, 'epoch: ' + str(epoch) + ' ' + txt)
    log(f, 'Exiting train...')
    f.close()
    return
##################################################################################

#################################################################


def test_classifier(network, dataset, snn_loaded_object):
    loss = 0.0
    accuracy = 0.0
    for i, (X, Y) in enumerate(dataset):

        _, loss_tmp, acc_tmp = network.test(get_gap(snn_loaded_object.get_final_voltage(X, 0.0)), Y)
        loss += loss_tmp
        accuracy += acc_tmp

    i += 1
    loss /= i
    accuracy /= i
    txt = 'Accuracy: %.4f%%, loss: %.12f, i: %d' % (accuracy * 100.0, loss, i)
    return loss, accuracy, txt

#####################################################################################


def train_classifer_main():

    np.random.seed(11)
    data_path = 'mnistdata/'
    batch_size = 1

    model_save_path = './mymodels'

    print 'Loading Dataset'
    mnist = mnist_data_set(data_path, batch_size)
    print 'done loading'

    datasets = mnist.data_sets
    print type(datasets['train'].X)
    print np.max(datasets['train'].X), np.min(datasets['train'].X)
    print np.max(datasets['test'].X), np.min(datasets['test'].X)
    print datasets['train'].X.shape, datasets['train'].Y.shape
    print datasets['test'].X.shape, datasets['test'].Y.shape
   
    data_shape = datasets['train'].X.shape
    data_shape = (batch_size, ) + data_shape[1:]
    print 'Data shape:', data_shape

    path = os.path.join(model_save_path, 'train2')

    print 'loading snn'
    f = open(os.path.join(path, 'snn_autonet' + '.save'), 'rb')
    snn_loaded_object = cPickle.load(f)
    f.close()
    print('Done')

    path = os.path.join(model_save_path, 'classifer_SGD01')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'snapshots'))

    # print ('creating classifer')
    # classifier = softmax_classifier()
    print('loading classifier')
    f = open(os.path.join(path, 'softmax_classifer' + '.save'), 'rb')
    classifier=cPickle.load(f)
    f.close()
    print('done')

    print('classifier TRAINING ...')
    train_classifer(classifier, datasets, path, snn_loaded_object)
    print('completed training classifer!')
    print('saving classifier...')
    f = open(os.path.join(path, classifier.name + '.save'), 'wb')
    # theano.misc.pkl_utils.dump()
    sys.setrecursionlimit(50000)
    cPickle.dump(classifier, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print('Done')


if __name__ == '__main__':
    # def plot_weights(self, file_path = '.', plot_id = 0,
    #                  max_subplots = 64, max_figures = 16, layer_id = -1,
    #                  figsize = (6, 6)):
    #     i = -1

    #     W=self.all_layers[-1].W

    #     plot_weights(W, 'ID' + str(plot_id) + '_' + 'snn', file_path,

    '''
    np.random.seed(11)
    
    model_save_path = './models'

    path = os.path.join(model_save_path, 'train1')

    print 'loading snn'
    f = open(os.path.join(path, 'snn_autonet' + '.save'), 'rb')
    network=cPickle.load(f)
    f.close()
    print('Done')

    batch_size = 1

    # snn_loaded_object.all_layers[-1].stdp_enabled=False

    data_path = '/data3/deepak_interns/vikram/Face_Mbike/'

    print 'Loading Dataset'
    mnist = mnist_data_set(data_path, batch_size)
    print 'done loading'

    datasets = mnist.data_sets
    print type(datasets['train'].X)
    print np.max(datasets['train'].X), np.min(datasets['train'].X)
    
    data_shape = datasets['train'].X.shape
    data_shape = (batch_size, ) + data_shape[1: ]
    print 'Data shape:', data_shape

    for i, (X,Y) in enumerate(datasets['train']):
        if(i==10):
            break
        test_func=network.get_final_voltage

        output=get_gap(test_func(X,0))
        # sio.savemat('volatge'+str(i),{'v'+str(i) : output})
        print output
        print output.shape'''

    train_classifer_main()
