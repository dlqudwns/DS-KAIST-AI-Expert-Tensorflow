import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from helper import initialize_plot

# 비용 함수
def cost_func(x=None, y=None):
    x = x or tf.placeholder(tf.float32, shape=[None, 1])
    y = y or tf.placeholder(tf.float32, shape=[None, 1])
    z = -1 * tf.sin(x * x) * tf.cos(3 * y * y) * tf.exp(-(x * y) * (x * y)) - tf.exp(-(x + y) * (x + y))
    return x, y, z

ax = initialize_plot(cost_func)

# 변수들 시작 지점
x_0 = 0.75
y_0 = 1.0

# x, y 변수 초기화
x_var = tf.Variable(x_0, [1], dtype=tf.float32)
y_var = tf.Variable(y_0, [1], dtype=tf.float32)

# 계산 그래프 만들기
cost = cost_func(x_var, y_var)[2]

# 어떤 learning rate가 가장 좋을까?
LR = 0.1
color = 'r'

ops = tf.no_op()

######################################
##      그라디언트 하강 ops 정의      ##
## x_assign_op, y_assign_op 정의하기 ##
######################################

# x_grad = tf.gradients(cost, x_var)[0]
# x_assign_op = tf.assign(x_var, x_var - LR * x_grad)
# y_grad = tf.gradients(cost, y_var)[0]
# y_assign_op = tf.assign(y_var, y_var - LR * y_grad)
# ops = tf.group([x_assign_op, y_assign_op])

# 혹은 아래 코드를 통해 간단하게 구현 가능
ops = tf.train.GradientDescentOptimizer(LR).minimize(cost)

# 시각화 코드
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 선을 그리기 위한 마지막 장소 저장
    last_x, last_y, last_z = [], [], []
    plot_cache = None

    # 그라디언트 하강을 100번 반복
    steps = 100
    for iter in range(steps):
        # x, y 변수들을 한번 갱신하고 그 값들 및 비용을 수집
        _, x_val, y_val, z_val = sess.run([ops, x_var, y_var, cost])

        # 점을 현재 값으로 이동
        if plot_cache:
            plot_cache[0].remove(), plot_cache[1].remove()
        print(x_val, y_val, z_val)
        plot_cache = [ax.scatter(x_val, y_val, z_val, s=3, depthshade=True, color=color),
                      ax.text(x_val, y_val , z_val, '  x: {:.2f}, y: {:.2f}, cost: {:.2f}'.format(x_val, y_val, z_val), fontsize=3)]

        # 마지막 값으로부터 선 그리기
        if iter == 0:
            last_x, last_y, last_z = x_0, y_0, z_val
        ax.plot([last_x, x_val], [last_y, y_val], [last_z, z_val], linewidth=0.5, color=color)
        last_x, last_y, last_z = x_val, y_val, z_val

        if iter == 0:
            plt.legend([plot_cache], [LR])

        print('iteration: {}'.format(iter))
        plt.pause(0.0001)

print("done")
plt.pause(100)
