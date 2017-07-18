'''
Created on 2017. 7. 17.
@summary: Chapter 3 : Clustering(군집화) - K 평균
@author: Byungchul
'''

'''
 tensor
 - Tensorflow의 기본 자료구조
 - 동적 크기를 갖는 다차원 데이터 배열

 rank(랭크)
 - length of tensor
 
 shape(구조)
 - tensor의 구조
 
 []는 랭크 0, 차원번호(Dimension number) 0-D
 [D0]는 랭크 1, 차원번호 1-D
 [D0, D1]는 랭크 2, 차원번호 2-D


* K-평균 알고리즘이란?

주어진 데이터를 군집(Cluster) 개수(K)로 그룹화
- 한 Cluster 내 데이터들은 동일한 성질

이 알고리즘의 결과는 Centroid(중심)이라고 부르는 K개의 점
모든 데이터는 K개의 Cluster 중 하나에 속한다

군집 구성 오차함수를 최소화하기 위해 반복 개선(Iterative Refinement) 기법을 쓴다.

초기단계(0단계) : K개 중심의 초기 집합을 결정
- 예제에서는 임의로 선택하여 결정
할당 단계(1단계) : 각 데이터를 가장 가까운 군집에 할당
업데이트 단계(2단계) : 각 그룹에 대해 새로운 중심을 계산
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
    초기단계(0단계)
'''
vectors_set = []

# 2000개의 2차원 좌표 랜덤 생성
for i in range(2000) : 
    if np.random.random() > 0.5 :
        vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])    # 평균이 0이고 표준편차가 0.9
    else :
        vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

# 그래프로 표현
plt.plot([v[0] for v in vectors_set], [v[1] for v in vectors_set], 'ro')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()

'''
    할당단계(1단계)
'''
# 모든 데이터를 텐서로 옮기자
vectors = tf.constant(vectors_set) # 상수 텐서

# 무작위의 k개의 중심을 선택
k = 4
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k,-1]))
print('최초 k개의 중심 : ', centroids)

# 유클리드 제곱 거리를 구해 가장 가까운 중심 계산
'''
tf.subtract(vectors, centroids)를 하려고 함. 하지만 두 텐서의 1차원의 크기가 다름(2000 vs 4)
그래서 expand_dims()를 함
vectors는 D0을 추가
centroids에는 D1을 추가

'''
expanded_vectors = tf.expand_dims(vectors,0)
expanded_centroids = tf.expand_dims(centroids,1)

print(expanded_vectors.get_shape())
print(expanded_centroids.get_shape())

# shape를 보면 크기가 1인 차원은 텐서 연산 시 다른 텐서의 해당 차원 크기에 맞게 알아서 계산을 반복!
diff = tf.subtract(expanded_vectors, expanded_centroids)
sqr = tf.square(diff)
distances = tf.reduce_sum(sqr, 2) # 지정된 차원을 따라 원소를 더함
assignments = tf.argmin(distances, 0) # 가장 작은 값의 원소가 있는 인덱스를 리턴

print('diff : ', diff.get_shape())
print('sqr : ', sqr.get_shape())
print('distances : ', distances.get_shape())
print('assignments : ', assignments.get_shape())

'''
    수정단계(2단계)
'''

means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])),
                                  reduction_indices=[1]) for c in range(k)], 0)


# 그래프 실행
update_centroids = tf.assign(centroids, means)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

print('first result : ', sess.run(centroids))

for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

print('result : ', centroid_values)

sess.close()










