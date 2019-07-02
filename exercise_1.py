import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filenames = ["./cat.jpg", "./dog.jpg"]
raw_images = [mpimg.imread(fn) for fn in filenames]
image = tf.placeholder("int32", [None, None, None])

with tf.Session() as session:

    # 이 부분을 수정
    preprocessed_image = image 
    
    # tf.reverse를 이용하여 이미지 뒤집기

    # tf.shape와 array slicing을 이용하여 이미지 사이즈 600x800으로 자르기
    # 참고: // 를 이용하여 int / int -> int로 (버림) true division이 가능함

    # 흑백 이미지로 만들기
    # 방법 1: tf.ones로 만든 1 matrix를 tf.matmul으로 행렬곱하고 3으로 나누어서 흑백 이미지 만들기
    # 이 과정에서 tf.expand_dims와 tf.squeeze가 필요함

    # 방법 2: tf.reduce_mean을 이용하여 흑백 이미지 만들기

    
    results = [session.run(
        preprocessed_image, feed_dict={image: ri}) for ri in raw_images]

for r in results:
    plt.imshow(r)
    plt.show()

