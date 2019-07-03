import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import initialize_plot_ex3

LR = 0.1   # 러닝 레이트: 낮으면 학습이 느리고, 높으면 발산의 위험이 있음. 0.1, 1.0을 넣어 확인 가능
REAL_PARAMS = [1.2, 2.5]   # 찾아야 할 정답 파라미터
INIT_PARAMS = [[5, 4], [5, 1], [2, 4.5]][2]   # 시작 파라미터. 0, 1, 2 를 넣어서 결과를 확인 가능
x = np.linspace(-1, 1, 200, dtype=np.float32)   # x 데이터

# 아래에서 원하는 함수의 코멘트를 풀어서 각각 종류의 함수를 테스트 가능

# 선형 함수
y_fun = lambda a, b: a * x + b
tf_y_fun = lambda a, b: a * x + b

# 다항 함수
# y_fun = lambda a, b: a * x**3 + b * x**2
# tf_y_fun = lambda a, b: a * x**3 + b * x**2

# 신경망 함수
# y_fun = lambda a, b: np.sin(b*np.cos(a*x))
# tf_y_fun = lambda a, b: tf.sin(b*tf.cos(a*x))

noise = np.random.randn(200)/10
y = y_fun(*REAL_PARAMS) + noise         # 타겟 데이터

ax1, ax2 = initialize_plot_ex3(y_fun, x, y)

# 텐서플로우 그래프
a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in INIT_PARAMS]
pred = tf_y_fun(a, b); mse = tf.reduce_mean(tf.square(y-pred))
train_op = tf.train.GradientDescentOptimizer(LR).minimize(mse)

# 그라디언트 계산 및 파라미터 갱신 op
a_list, b_list, cost_list = [], [], []
plot_cache = None
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(100):
        a_, b_, mse_ = sess.run([a, b, mse])
        a_list.append(a_); b_list.append(b_); cost_list.append(mse_)    # 파라미터 변화 저장
        result, _ = sess.run([pred, train_op])                          # 트레이닝
        if plot_cache:
            plot_cache[0].remove(); plot_cache[1].remove()
        plot_cache = [ax1.plot(x, result, 'r-', lw=2)[0],
                      ax2.text(a_, b_, mse_,
                              '  a: {:.2f}, b: {:.2f}, cost: {:.2f}'.format(a_, b_, mse_),
                              fontsize=10)]
        if t == 0:
            ax2.scatter(a_, b_, zs=mse_, s=300, c='r')
        ax2.plot(a_list[-2:], b_list[-2:], cost_list[-2:], zdir='z', c='r', lw=3)
        plt.pause(0.1)

ax2.scatter(a_list[-1], b_list[-1], zs=cost_list[-1], s=300, marker = '*', c='r')

print("done")
print('a=', a_, 'b=', b_)
plt.pause(100)
