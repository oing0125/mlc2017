'''
Created on 2017. 7. 15.

@author: Byungchul
'''
import tensorflow as tf
from numpy import square

# 고정값을 가진 node
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

print(node1, node2)

# Session을 통한 run으로 evaluate node
sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3 : ", node3)
print("node3 : ", sess.run(node3))

# 가변 값은 placeholders를 쓴다
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)


####################### tensor flow 기본 연산 ###########################

# a + b는 tf.add(a,b)와 같다
adder_node = a + b

print(sess.run(adder_node, {a:3, b:4.5}))
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a:[3,4], b:[4.5,1]}))

########################### linear model ###############################

# Variables를 이용해서 trainable parameter를 graph에 더할 수 있다.
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# init을 안하면 전역 변수로 선언된 variables를 쓸 수가 없다.
init = tf.global_variables_initializer()
# 아래 코드로 선언된 variables를 초기화
sess.run(init)

print(W.eval(sess))
print(b.eval(sess))
print(sess.run(linear_model, {x:[1,2,3,4]}))


# 만든 모델을 평가
# y는 만든 model에 x를 넣었을 때 얻기를 원하는 값
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
# loss를 통해 현재 model이 예상치에 얼마나 떨어져있는지 알 수 있다.
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# loss value를 줄이기 위해 W와 b를 재설정하여 예상치로 접근할 수 있다.
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print("after fix X, b", sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


#TODO https://www.tensorflow.org/get_started/get_started
########################### train API #############################














sess.close()



