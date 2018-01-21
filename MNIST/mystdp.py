# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import timeit
#import pickle
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


#from mnist_reader import data_set, mnist_data_set

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

from theano import Apply
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
# from theano.sandbox.cuda import CudaNdarray
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, gpu_contiguous


def relu1(x):
    return T.switch(x < 0, 0, x)


class stdpOp(GpuOp):

    __props__ = ()

    def __init__(self):

        self.max_threads_dim0 = None
    # 这里的output_spike是卷积层输出的脉冲序列，这里其实是作为apply的INPUT

    def make_node(self, output_spike, H_out, weights):
        # 该函数是为了构造apply节点以及连着它的相关节点
        if output_spike.type.ndim != 4:  # conv1 (1,20,28,28)
            raise TypeError('output_spike must be 4D tensor')
        if H_out.type.ndim != 4:  # conv1 (1,2,28,28)
            raise TypeError('H_out must be 4D tensor')
        if weights.type.ndim != 4:  # conv1 (30,2,5,5)
            raise TypeError('weights must be 4D tensor')
        # if LR.type.ndim != 1:
        #     raise TypeError('LR must be 1D tensor')
        # if weight_update.type.ndim != 4:
        #     raise TypeError('weight_update must be 4D tensor')

        output_spike = as_cuda_ndarray_variable(output_spike, )
        H_out = as_cuda_ndarray_variable(H_out)
        weights = as_cuda_ndarray_variable(weights)
        # LR= as_cuda_ndarray_variable(LR)
        # weight_update = as_cuda_ndarray_variable(weight_update)

        print 'MAKENODE: ', output_spike.shape, H_out.shape, weights.shape
        # broadcastable = [output_spike.type.broadcastable[0], H_out.type.broadcastable[0],
        # weights.type.broadcastable[0], weight_update,False, False, False, False]
        # otype = CudaNdarrayType(broadcastable=[False] * 4)
        broadcastable = [False, False, False, False, False]
        # 后面是创建一个输出变量，类型是CudaNdarrayType，使能广播。 [output_var]
        return Apply(self, [output_spike, H_out, weights], [CudaNdarrayType(broadcastable)()])

    def prepare_node(self, node, storage_map, compute_map, impl):
        super(stdpOp, self).prepare_node(node, storage_map, compute_map, impl)
        print 'IN PREPARE NODE\n'
        if node.op.max_threads_dim0 is None:
            cuda = theano.sandbox.cuda
            device_id = cuda.use.device_number
            if device_id is None:
                cuda.use("gpu",
                         force=False,
                         default_to_move_computation_to_gpu=False,
                         move_shared_float32_to_gpu=False,
                         enable_cuda=False,
                         test_driver=True)
                device_id = cuda.use.device_number
            cuda_ndarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
            prop = cuda_ndarray.device_properties(device_id)
            node.op.max_threads_dim0 = prop['maxThreadsDim0']

    def c_headers(self):
        return ['cuda_ndarray.cuh', '<stdio.h>']

    '''
    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (0, 23)
    '''

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        files = ['mystdp_kernel.cu']
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read() for f in files]

        return reduce(str.__add__, codes)
        # codes 是一个列表，str.__add__是一个函数，必须从列表中选择两个参数传入
        # reduce()对list的每个元素反复调用函数f，并返回最终结果值。 str函数会返回一个对象的str格式

    def c_code(self, node, nodename, inp, out_, sub):
        '''
        node 是对一个apply节点的引用，从make_node()方法获得。
        inp 是一系列字符串，作为op的输入，每个字符串都包含了与输入一致的C变量名字，C变量的名字代表了inp[0]给的第一个输入，
        要用这个名字去处理这个变量。out_ 与inp意义一致。
        sub 是一个额外参数的字典，包括sub['fail']，用于触发异常
        This method returns a string containing the C code to perform the computation required by this op.
        '''
        output_spike, H_out, weights = inp
        out, = out_
        max_threads_dim0 = self.max_threads_dim0

        sub = sub.copy()  # 浅复制
        # sub is a dictionary of extras parameters to the c_code method.Among other things, it contains sub['fail']
        sub.update(locals())  # locals()提供基于字典的访问局部变量的方式，局部名字空间
        # 每个函数都有着自已的名字空间，叫做局部名字空间，它记录了函数的变量，包括函数的参数
        # 和局部定义的变量。每个模块拥有它自已的名字空间，叫做全局名字空间，它记录了模块的变
        # 量，包括函数、类、其它导入的模块、模块级的变量和常量。还有就是内置名字空间，任何模
        # 块均可访问它，它存放着内置的函数和异常。locals()只读
        # print('hello')
        # exit(0)

        # PyErr_Format() supports string formatting so it is possible to tailor the error message to the specifics
        # of the error that occured. 以下c代码字符串为一个template，进行 template % sub， 字典的格式化字符串
        # fprintf 格式化输出到一个流/文件中
        # 宏Py_XINCREF和Py_XDECREF用来增加和减少对象的引用次数
        # PyObject * CudaNdarray_ZEROS(int n, int * dims)
        return """

    const int *os_size = CudaNdarray_HOST_DIMS(%(output_spike)s);
    const int *h_size = CudaNdarray_HOST_DIMS(%(H_out)s);
    const int *w_size = CudaNdarray_HOST_DIMS(%(weights)s);
    int delta_w_size[5] = {os_size[0], w_size[0], w_size[1], w_size[2], w_size[3]};
    int cpu_Ron[1500] = 10;
    long cpu_Roff[1500] = 420;
    float cpu_vol[1500] = 1.65;
    float cpu_m[1500] = -0.13;
    float cpu_n[1500] = 0.28;
    float cpu_p[1500] = 0.48;
    float cpu_q[1500] = 0.165;
    float cpu_pp[1500] = 1.54;
    float cpu_qq[1500] = 0.54;
    long cpu_a[1500] = 300;
    int cpu_b[1500] = -100;
    float cpu_dt[1500] = 0.001;
         
    __constant__ const int gpu_Ron[1500];
    __constant__ const long gpu_Roff[1500];
    __constant__ const float gpu_vol[1500];
    __constant__ const float gpu_m[1500];
    __constant__ const float gpu_n[1500];
    __constant__ const float gpu_p[1500];
    __constant__ const float gpu_q[1500];
    __constant__ const float gpu_pp[1500];
    __constant__ const float gpu_qq[1500];
    __constant__ const long gpu_a[1500];
    __constant__ const int gpu_b[1500]
    __constant__ const float gpu_dt[1500];
    
    cudaMemcpyToSymbol(gpu_Ron, cpu_Ron, 1500 * sizeof(int));
    cudaMemcpyToSymbol(gpu_Roff, cpu_Roff, 1500 * sizeof(long));
    cudaMemcpyToSymbol(gpu_vol, cpu_vol, 1500 * sizeof(float));
    cudaMemcpyToSymbol(gpu_m, cpu_m, 1500 * sizeof(float));
    cudaMemcpyToSymbol(gpu_n, cpu_n, 1500 * sizeof(float));
    cudaMemcpyToSymbol(gpu_p, cpu_p, 1500 * sizeof(float));
    cudaMemcpyToSymbol(gpu_q, cpu_q, 1500 * sizeof(float));
    cudaMemcpyToSymbol(gpu_pp, cpu_pp, 1500 * sizeof(float));
    cudaMemcpyToSymbol(gpu_qq, cpu_qq, 1500 * sizeof(float));
    cudaMemcpyToSymbol(gpu_a, cpu_a, 1500 * sizeof(long));
    cudaMemcpyToSymbol(gpu_b, cpu_b, 1500 * sizeof(int));
    cudaMemcpyToSymbol(gpu_dt, cpu_dt, 1500 * sizeof(float));
    
    if (os_size[1] > %(max_threads_dim0)s)
    {
        fprintf(stderr, "\\nSTDP_OP ERROR: CHANNEL SIZE EXCEEDED THREAD LIMIT (%%d).\\n", %(max_threads_dim0)s);
    }

    Py_XDECREF(%(out)s);

    %(out)s = (CudaNdarray*)CudaNdarray_ZEROS(5,delta_w_size);  //zeros uses int* while ndims uses const int * as second argument
    if (NULL == %(out)s)
    {
        PyErr_Format(PyExc_RuntimeError,
                    "stdpOpMM: Failed to allocate output of %%d x %%d x %%d x %%d",
                    w_size[0], w_size[1], w_size[2], w_size[3]);
    }

    if (!(CudaNdarray_is_c_contiguous(%(output_spike)s) && CudaNdarray_is_c_contiguous(%(H_out)s) \
            && CudaNdarray_is_c_contiguous(%(weights)s) && CudaNdarray_is_c_contiguous(%(out)s)))
    {
        fprintf(stderr, "\\nSTDP_OP ERROR: VARIABLES NOT C-CONTIGUOUS.\\n");
    }

    //dim3 threads(threadx,thready);
    int threads = os_size[1];
    dim3 grid(os_size[0], os_size[2], os_size[3]);

    stdp_kernel <<< grid, threads >>> (%(weights)s->devdata, w_size[0], w_size[1], w_size[2], w_size[3], 
                                        %(output_spike)s->devdata, os_size[0], os_size[1], os_size[2], os_size[3],
                                        gpu_Ron, gpu_Roff, gpu_vol, gpu_m, gpu_n, gpu_p, gpu_q,
                                        gpu_pp,gpu_qq, gpu_a, gpu_b, gpu_dt, 
                                        %(H_out)s->devdata, %(out)s->devdata);
    CNDA_THREAD_SYNC;
    cudaError_t sts = cudaGetLastError();
    if (cudaSuccess != sts)
    {
        fprintf(stderr, "\\nSTDP_OP KERNEL ERROR: error_code=%%d, %%s.\\n", sts, cudaGetErrorString(sts));
    }

    //Py_XDECREF(%(out)s);
    if (%(out)s == NULL)
    {
        %(fail)s
    }
""" % sub


if __name__ == '__main__':
    import numpy as np

    # dict=sio.loadmat('conv_op_test.mat')
    #
    # w=dict('w')
    # os=dict('os')
    # h_out=dict('h_out')
    a = theano.tensor.tensor4()
    b = theano.tensor.tensor4()
    c = theano.tensor.tensor4()
    f = theano.function([a, b, c], stdpOp()(a, b, c))
    print 'compiled'
    # exit(0)
    
    x = 1.0*np.ones((32, 30, 300, 400), dtype=np.float32)
    y = 1.0*np.ones((32, 2, 300, 400), dtype=np.float32)
    z = 0.5*np.ones((30, 2, 10, 10), dtype=np.float32)
    print 'before', datetime.datetime.now()
    out = f(x, y, z)
    print 'after', datetime.datetime.now()
    print 'out shape', out.shape
    print 'out', out
    print 'computed happily!!!'
