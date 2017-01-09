import tensorflow as tf

with tf.device('/gpu:0'):
    rand_var_1 = tf.Variable(tf.random_uniform([512, 512], 0, 10, dtype=tf.int32, seed=0))
    rand_var_2 = tf.Variable(tf.random_uniform([512, 512], 0, 10, dtype=tf.int32, seed=0))
    product = tf.matmul(rand_var_1, rand_var_2)

# Launch the default graph.
sess = tf.Session()
sess.run(tf.initialize_all_variables())
while True:
    result = sess.run(product)
print(result)
# ==> [[ 12.]]

# Close the Session when we're done.
sess.close()
