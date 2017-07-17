'''
Created on 2017. 7. 17.
@summary: Chapter 2 : Linear Regression(선형회귀분석)
@author: Byungchul
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# y = 0.1 * x + 0.3을 만들어보자
num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
#     y1 = x1 * 0.1 + 0.3 
    # 아래 코드의 의미는 정규분포를 따르는 약간의 변동값을 더해 직선과 완전히 일치하지 않게 함.
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# matplotlib 패키지로 도식화
plt.plot(x_data, y_data, 'ro')
# plt.show()


"""
    x_data로  y 출력값을 추정하는 알고리즘을 만들자
    y_data = W * x_data + b 공식에서 W와 b를 찾아보자
"""

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# x_data와 y_data로 실제 값과 위 공식으로 계산한 값 사이의 거리를 기반으로 비용함수를 만든다.
# loss = 거리의 제곱한 것의 합계를 평균
loss = tf.reduce_mean(tf.square(y - y_data))

# Gradient Descent 알고리즘으로 찾아보자
# 초기 시작점에서 함수의 값이 최소가 되게 하면서 반복적으로 최적화를 수행
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 0.5는 학습속도, 뒤에 설명함
train = optimizer.minimize(loss)

# 알고리즘 실행을 위한 초기화
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# 알고리즘 실행
for i in range(20) :
    # W와 b의 변화를 알기 위해 출력해보자
    sess.run(train)
    print(i, sess.run(W), sess.run(b), sess.run(loss))

print(sess.run(W), sess.run(b))

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.xlabel('x')
plt.ylabel('y')
plt.show()

"""
    Gradient Descent는 오차함수의 기울기를 계산하기 위해 오차함수를 미분
    optimizer에서 학습속도는 텐서플로가 각 반복마다 얼마나 크게 이동할 것인자를 정할 수 있다
"""