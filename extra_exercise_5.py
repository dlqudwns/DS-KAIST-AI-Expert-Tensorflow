import tensorflow as tf, numpy as np

# 데이터셋 다운로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train, x_test, y_test = x_train[y_train < 2], y_train[y_train < 2], x_test[y_test < 2], y_test[y_test < 2]

# 텐서 모양을 잘 맞춰주기

# 파라미터 및 변수 정의

# 비용함수 및 갱신 연산 정의
prediction = tf.nn.sigmoid(tf.matmul(X, W) + b)
cost = tf.losses.sigmoid_cross_entropy(Y, prediction)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.round(prediction), Y), tf.float32))

train_op = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost) # 적절한 learning rate의 값은?

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
    
    ### 훈련 코드 필요한 부분 ###
    ###                     ###
    ###########################

  print("Test accuracy", sess.run(acc, feed_dict={X: x_test, Y: y_test}))
