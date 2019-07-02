import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LR = 0.1   # 러닝 레이트: 낮으면 학습이 느리고, 높으면 발산의 위험이 있음. 0.1, 1.0을 넣어 확인 가능
REAL_PARAMS = [1.2, 2.5]   # 찾아야 할 정답 파라미터
INIT_PARAMS = [[5, 4], [5, 1], [2, 4.5]][2]   # 시작 파라미터. 0, 1, 2 를 넣어서 결과를 확인 가능
x = np.linspace(-1, 1, 200, dtype=np.float32)   # x 데이터

# 아래에서 원하는 함수의 코멘트를 풀어서 각각 종류의 함수를 테스트 가능

# 선형 함수
y_fun = lambda a, b: a * x + b
tf_y_fun = lambda a, b: a * x + b

# 다항 함수
#y_fun = lambda a, b: a * x**3 + b * x**2
#tf_y_fun = lambda a, b: a * x**3 + b * x**2

# 신경망 함수
#y_fun = lambda a, b: np.sin(b*np.cos(a*x))
#tf_y_fun = lambda a, b: tf.sin(b*tf.cos(a*x))

noise = np.random.randn(200)/10
y = y_fun(*REAL_PARAMS) + noise         # 타겟 데이터

# 텐서플로우 그래프
a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in INIT_PARAMS]
pred = tf_y_fun(a, b); mse = tf.reduce_mean(tf.square(y-pred))
train_op = tf.train.GradientDescentOptimizer(LR).minimize(mse)    
# 그라디언트 계산 및 파라미터 갱신 op
a_list, b_list, cost_list = [], [], []
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for t in range(400):
    a_, b_, mse_ = sess.run([a, b, mse])
    a_list.append(a_); b_list.append(b_); cost_list.append(mse_)    # 파라미터 변화 저장
    result, _ = sess.run([pred, train_op])                          # 트레이닝

# 시각화 코드:
print('a=', a_, 'b=', b_)
plt.figure(1); plt.scatter(x, y, c='b')    # 데이터 플롯
plt.plot(x, result, 'r-', lw=2)   # 라인 피팅 플롯
fig = plt.figure(2); ax = Axes3D(fig) # 3D 비용함수 플롯
a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))  # 파라미터 공간
cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y)) for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)  # 파라미터 공간에서의 비용함수 계산
ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5)
ax.scatter(a_list[0], b_list[0], zs=cost_list[0], s=300, c='r')  # 시작 파라미터 지점
ax.set_xlabel('a'); ax.set_ylabel('b')
ax.plot(a_list, b_list, zs=cost_list, zdir='z', c='r', lw=3)    # 3D 그라디언트 하강 플롯
ax.scatter(a_list[-1], b_list[-1], zs=cost_list[-1], s=300, marker = '*', c='r')  # 마지막 파라미터
plt.show()

