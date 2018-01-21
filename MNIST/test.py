# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

# rng = np.random
# # Training data
# N = 400
# feats = 784
# D = (rng.randn(N, feats).astype(theano.config.floatX),
#      rng.randint(size=N,low=0, high=2).astype(theano.config.floatX))
# training_steps = 10000
# # Declare Theano symbolic variables
# x = T.matrix("x")
# y = T.vector("y")
# w = theano.shared(rng.randn(feats).astype(theano.config.floatX), name="w")
# b = theano.shared(np.asarray(0., dtype=theano.config.floatX), name="b")
# x.tag.test_value = D[0]
# y.tag.test_value = D[1]
# # Construct Theano expression graph
# p_1 = 1 / (1 + T.exp(-T.dot(x, w)-b))  # Probability of having a one
# prediction = p_1 > 0.5  # The prediction that is done: 0 or 1
# # Compute gradients
# xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1)  # Cross-entropy
# cost = xent.mean() + 0.01*(w**2).sum()  # The cost to optimize
# gw, gb = T.grad(cost, [w, b])
# # Training and prediction function
# train = theano.function(inputs=[x, y], outputs=[prediction, xent],
#                         updates=[[w, w-0.01*gw], [b, b-0.01*gb]], name="train")
# predict = theano.function(inputs=[x], outputs=prediction, name="predict")
#
# theano.printing.pprint(prediction)
# 'gt((TensorConstant{1} / (TensorConstant{1} + exp(((-(x \\dot w)) - b)))), TensorConstant{0.5})'
#
# #theano.printing.debugprint(prediction)  # 预编译的图及图片打印
# theano.printing.pydotprint(prediction, outfile="pics/logreg_pydotprint_prediction.png", var_with_name_simple=True)
#
# theano.printing.debugprint(predict)   # 编译后的图及图片打印
# theano.printing.pydotprint(predict, outfile="pics/logreg_pydotprint_predict.png", var_with_name_simple=True)
#
# theano.printing.debugprint(train)
# theano.printing.pydotprint(train, outfile="pics/logreg_pydotprint_train.png", var_with_name_simple=True)
#
# # x= T.dmatrix('x')
# # y= x * 2.
# # >>>y.owner.op.name
# # 'Elemwise{mul,no_inplace}'#y的owner是apply而apply的op是'Elemwise{mul,no_inplace}'
# # >>>len(y.owner.inputs)
# # 2#两个输入
# # >>>y.owner.inputs[0]
# # x#第一个输入是x矩阵
# # >>>y.owner.inputs[1]
# # InplaceDimShuffle{x,x}.0
# # 注意这里的第二个输入并不是2，而是和x同样大小的矩阵框架，要广播才能相乘
#
# # theano.scan 中fn的输入变量顺序为sequences中的变量，outputs_info的变量，non_sequences中的变量。如果使用了taps，则按照taps给fn喂变量.


#  scan 1 ================================================================================
# k = T . iscalar('k')
# A = T . vector('A')
#
# outputs, updates = theano.scan(lambda result, A: result * A,
#                                non_sequences=A, outputs_info=T.ones_like(A), n_steps=k)
# result = outputs[-1]
# fn_Ak = theano . function([A, k], result, updates=updates )
# print fn_Ak(range(10), 2)
# #  scan 2 ================================================================================
# # polynomial -- c0*x^0 + c1*x^1 + c2*x^2 + c3*x^3...
# # ========================================================================================
# coefficients = T.vector('coeff')
# x = T.iscalar('x')
# sum_poly_init = T.fscalar('sum_poly')
# result, update = theano.scan(lambda coefficients, power, sum_poly, x: T.cast(sum_poly +
#                              coefficients*(x**power), dtype=theano.config.floatX),
#                              sequences=[coefficients, T.arange(coefficients.size)],
#                              outputs_info=[sum_poly_init],
#                              non_sequences=[x])
#
# poly_fn = theano.function([coefficients, sum_poly_init, x], result, updates=update)
#
# coeff_value = np.asarray([1.,3.,6.,5.], dtype=theano.config.floatX)
# x_value = 3
# poly_init_value = 0.
# print poly_fn(coeff_value,poly_init_value, x_value)
# # scan会在T.arange()生成的list上遍历。例如这段代码中T.arange()生成list=[0,1,2,3]，在第i次迭代中，
# # scan把coefficients的第i个元素和list的第i个元素喂给fn作为参数。outputs_info作为第三个参数输入给fn，
# # 然后是non_sequences的变量。其中outputs_info的初始化大小和类型都要和fn的返回结果相同。
# # 打印结果中包含了4次迭代的输出。
#
# #  scan 3 ================================================================================
# # theano.scan_module.until
# # ========================================================================================
# print 'theano.scan_module.until:'
#
#
# def prod_2(pre_value, max_value):
#
#     return pre_value*2, theano.scan_module.until(pre_value*2 > max_value)
#
# max_value = T.iscalar('max_value')
# result, update = theano.scan(prod_2, outputs_info=[T.constant(1.), T.constant(2.)],
#                              non_sequences=[max_value], n_steps=100)
#
# prod_fn = theano.function([max_value], result, updates=update)
# print prod_fn(400)
#
# #  scan 4 ================================================================================
# #  如果有如下代码
# #  scan(fn, sequences = [ dict(input= Sequence1, taps = [-3,2,-1])
# #                      , Sequence2
# #                      , dict(input =  Sequence3, taps = 3) ]
# #        , outputs_info = [ dict(initial =  Output1, taps = [-3,-5])
# #                         , dict(initial = Output2, taps = None)
# #                         , Output3 ]
# #        , non_sequences = [ Argument1, Argument2])
# #  那么fn函数的变量输入顺序是,t可以视为当前的迭代时间数
# #    1. Sequence1[t-3]
# #    2. Sequence1[t+2]
# #    3. Sequence1[t-1]
# #    4. Sequence2[t]
# #    5. Sequence3[t+3]
# #    6. Output1[t-3]
# #    7. Output1[t-5]
# #    8. Output3[t-1]
# #    9. Argument1
# #    10. Argument2
# #  其中sequences默认taps=0；outputs_info默认taps=-1，因为taps=0的结果是当前这一步迭代需要计算的。
# # ========================================================================================
# # taps scalar -- Fibonacci sequence
# # ========================================================================================
# Fibo_arr = T.vector('Fibonacci')
# k = T.iscalar('n_steps')
# result, update = theano.scan(lambda tm2, tm1: tm2 + tm1,
#                              outputs_info=[dict(initial=Fibo_arr, taps=[-2, -1])], n_steps=k)
# Fibo_fn = theano.function([Fibo_arr, k], result, updates=update)
# Fibo_init = np.asarray([1, 1], dtype=theano.config.floatX)
# k_value = 12
# print Fibo_fn(Fibo_init, k_value)

# 测试类函数的重载
# class ff(object):
#     def confuse(self):
#         print 'me too'
#         return 'c'
#
#
# class fff(ff):
#     def __init__(self):
#         self.a = 'a'
#         c ='c'
#         super(fff, self).__init__()
#         print 'i am a:'
#
#     def confuse(self):
#         b = 'b'
#         c = self.a+super(fff, self).confuse()
#         self.c = c
#         print 'i am confused'
#         return 'd'
#
# f = fff()
# print f.confuse()
# print f.c


# theano  dimshuffle调试
# import theano
# import theano.tensor as T
# import numpy as np
#
#
# a = np.array([[[[6,2,8],[1,4,5],[9,3,7]],[[1,9,5],[4,7,2],[8,3,6]]]])
# shape = a.shape
# x = theano.shared(a)  # 只有共享变量才能dimshuffle
# print 'a:', a
# threshold = 4
# refractory_voltage = -np.float32(10000000)
# amax, arg_max = T.max_and_argmax(x, axis=1, keepdims=True)
# print 'amax:', amax.eval()
# temp2 = T.signal.pool.pool_2d(amax, ws=(3, 3), ignore_border=True, stride=(1, 1), pad=(1, 1), mode='max', )
# print 'temp2:', temp2.eval()
# temp3 = T.reshape(T.switch(T.eq(temp2, x), x, 0.0), (shape[0], shape[1], -1))
# print 'temp3:', temp3.eval()
# a_spatial, a_spatial_argmax = T.max_and_argmax(temp3, axis=2, keepdims=True)
# print 'a_s', a_spatial.eval()
# print 'asa:', a_spatial_argmax.eval()
# thresh1 = T.gt(a_spatial, threshold).astype(theano.config.floatX)
# print 'thresh1:', thresh1.eval()
# thresh2 = T.gt(T.reshape(temp2, (shape[0], 1, -1)), threshold).astype(theano.config.floatX)
# print 'thresh2:', thresh2.eval()
# output = T.reshape((T.eq(T.arange(shape[2] * shape[3]).dimshuffle('x', 'x', 0), a_spatial_argmax) * thresh1), (shape[0], shape[1], shape[2], shape[3]))
# print 'output:', output.eval()
# flag = T.ge(thresh1 + thresh2, 1.0)   # 确定每个通道是否发放脉冲
# print flag.eval()
# temp4 = T.reshape(T.switch(flag, refractory_voltage, T.reshape(a, (shape[0], shape[1], -1))), (shape[0], shape[1], shape[2], shape[3]))
# print temp4.eval()
# # ========================================================================================

# 查看result到底是个什么东西
def test(a, b, c):
    A = T.cast(a*c, dtype=theano.config.floatX)
    B = T.cast(b*c, dtype=theano.config.floatX)
    return A, B,

max_value = T.iscalar('max_value')
result, update = theano.scan(test, outputs_info=[T.constant(1.), T.constant(2.)],
                             non_sequences=[max_value], n_steps=5)

prod_fn = theano.function([max_value], result, updates=update)
dd = prod_fn(400)
print dd
