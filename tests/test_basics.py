import pytest
import tensorflow as tf
from my_optimizer import MyOptimizer



def test_optimization():
    with tf.device('/cpu:0'):
        find_parabola(tf.optimizers.Adam())
        find_parabola(MyOptimizer())


def find_parabola(optimizer):
    with tf.device('/cpu:0'):
        var = tf.Variable(0.0)
        assert find_min_parabola_adam(var, optimizer).numpy() == pytest.approx(2, 0.1)
        print(var.numpy())
    

# @tf.function
def find_min_parabola_adam(var, optimizer):
    loss = lambda: (var-1.0)*(var-3.0)
    for _ in range(4000):
        optimizer.minimize(loss, [var])
    return var