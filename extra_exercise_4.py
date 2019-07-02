import tensorflow as tf

# 데이터셋 다운로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
y_train, y_test = y_train[:, None], y_test[:, None]

# 파라미터 및 변수 정의
with tf.variable_scope("parameters", reuse=tf.AUTO_REUSE):
  W = tf.get_variable("W", shape=(13, 1), initializer=tf.glorot_uniform_initializer())
  b = tf.get_variable("b", shape=(1, 1), initializer=tf.glorot_uniform_initializer())
X, Y = tf.placeholder(tf.float32, name="X"), tf.placeholder(tf.float32, name="Y")

# 비용함수 및 갱신 연산 정의
cost = tf.reduce_mean(tf.square(tf.matmul(X, W) + b - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3) # 적절한 learning rate의 값은?
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    ### 훈련 코드 필요한 부분 ###
    ###                     ###
    ###########################

    print("Test loss", sess.run(cost, feed_dict={X: x_test, Y: y_test}))
    # 예측이 정확한가 확인
    for i in range(10):
        print("x_data: {}, y_true: {}, y_pred: {}".format(
            x_test[i:i+1], y_test[i].item(), 
            sess.run(tf.matmul(X, W) + b, 
            feed_dict={X: x_test[i:i+1], Y: y_test[i:i+1]}).item()))
