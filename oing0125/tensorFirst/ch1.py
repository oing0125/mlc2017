'''
Created on 2017. 7. 17.

@author: Byungchul
'''
import tensorflow as tf

# placeholder : 실행 중에 값을 변경할 수 있음
a = tf.placeholder('float')
b = tf.placeholder('float')

# 곱하는 수학함수
# 자세한 주소는(https://goo.gl/sUQy7M)
y = tf.multiply(a, b)

# 수식을 계산하기 위한 Session
# Session을 생성함으로 텐서플로 라이브러리와 상호작용
sess = tf.Session()

print(sess.run(y, feed_dict={a : 3, b : 3}))


# Tensorflow는 알고리즘을 먼저 기술 => Session을 통한 연산 실행 구조

############## Tensorflow 용어 #####################
"""

Tensorflow 용어

그래프 구조 : 수학 계산
노드 : 수학 연산, 각종 처리
에지 : 입력 값과 출력 값으로 연결된 노드 사이의 관계, 동시에 Tensorflow의 기본 자료구조인 Tensor를 운반!

"""