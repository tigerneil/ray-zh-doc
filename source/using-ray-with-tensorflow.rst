结合 TensorFlow 使用 Ray
=========================

本文档描述结合 TensorFlow 使用 Ray 的最佳实践.

想要看到更多相关的样例，参看 `hyperparameter optimization`_,
`A3C`_, `ResNet`_, `Policy Gradients`_, and `LBFGS`_.

.. _`hyperparameter optimization`: http://ray.readthedocs.io/en/latest/example-hyperopt.html
.. _`A3C`: http://ray.readthedocs.io/en/latest/example-a3c.html
.. _`ResNet`: http://ray.readthedocs.io/en/latest/example-resnet.html
.. _`Policy Gradients`: http://ray.readthedocs.io/en/latest/example-policy-gradient.html
.. _`LBFGS`: http://ray.readthedocs.io/en/latest/example-lbfgs.html


如果你在分布式的设置上训练一个深度神经网络，那么你可能需要将网络在进程（或者机器）间进行传递. 比如说，你可能在一台机器上更新模型然后使用这个模型在另外一台机器上计算梯度. 但是，传递模型过程并不总是那么直接.

例如，直接尝试去 pickle 一个 TensorFlow 的计算图（graph）会得到一个混合的结果（mixed results）. 有些样例失败，有些例子会成功（但是会产生非常大的字符串）. 这些结果其实在其他 pickling 的库中也很类似.

另外，创建 TensorFlow 计算图需要数十秒的时间，所以序列一个图，并在另一个进程中重新创建这个图也非常低效. 更好的解决方法是开始时就在每个 worker 上创建同样的 TensorFlow 图，然后只需要将权重在 worker 间进行传递.

假设我们有一个简单网络的定义（这个是从 TensorFlow 文档中修改的版本）.

.. code-block:: python

  import tensorflow as tf
  import numpy as np

  x_data = tf.placeholder(tf.float32, shape=[100])
  y_data = tf.placeholder(tf.float32, shape=[100])

  w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
  b = tf.Variable(tf.zeros([1]))
  y = w * x_data + b

  loss = tf.reduce_mean(tf.square(y - y_data))
  optimizer = tf.train.GradientDescentOptimizer(0.5)
  grads = optimizer.compute_gradients(loss)
  train = optimizer.apply_gradients(grads)

  init = tf.global_variables_initializer()
  sess = tf.Session()

为了抽取权重和设置权重，你可以使用下面的帮助方法.

.. code-block:: python

  import ray
  variables = ray.experimental.TensorFlowVariables(loss, sess)

``TensorFlowVariables`` 对象提供方法来 get 和 set 权重并收集模型中所有的变量.

现在我们可以使用这些方法来抽取权重， 并将他们放回入网络中.

.. code-block:: python

  # First initialize the weights.
  sess.run(init)
  # Get the weights
  weights = variables.get_weights()  # Returns a dictionary of numpy arrays
  # Set the weights
  variables.set_weights(weights)

**注意：** 如果我们想要使用如下的方法 ``assign``设置权重，每次对 ``assign`` 的调用将会添加节点到图上，这个图随着时间不可控地变大.

.. code-block:: python

  w.assign(np.zeros(1))  # This adds a node to the graph every time you call it.
  b.assign(np.zeros(1))  # This adds a node to the graph every time you call it.

完整的样例
----------------

将这些放在一起，我们首先会将图放入到 Actor 中. 在 Actor 内部，我们会使用 ``TensorFlowVariables`` 类的 ``get_weights`` 和 ``set_weights`` 方法. 然后使用这些方法来在进程中进行权重传递（作为变量名映射到 numpy 数组的字典）而不需要传递是更加复杂的 Python 对象的实际的 TensorFlow 计算图.

.. code-block:: python

  import tensorflow as tf
  import numpy as np
  import ray

  ray.init()

  BATCH_SIZE = 100
  NUM_BATCHES = 1
  NUM_ITERS = 201

  class Network(object):
      def __init__(self, x, y):
          # Seed TensorFlow to make the script deterministic.
          tf.set_random_seed(0)
          # Define the inputs.
          self.x_data = tf.constant(x, dtype=tf.float32)
          self.y_data = tf.constant(y, dtype=tf.float32)
          # Define the weights and computation.
          w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
          b = tf.Variable(tf.zeros([1]))
          y = w * self.x_data + b
          # Define the loss.
          self.loss = tf.reduce_mean(tf.square(y - self.y_data))
          optimizer = tf.train.GradientDescentOptimizer(0.5)
          self.grads = optimizer.compute_gradients(self.loss)
          self.train = optimizer.apply_gradients(self.grads)
          # Define the weight initializer and session.
          init = tf.global_variables_initializer()
          self.sess = tf.Session()
          # Additional code for setting and getting the weights
          self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
          # Return all of the data needed to use the network.
          self.sess.run(init)

      # Define a remote function that trains the network for one step and returns the
      # new weights.
      def step(self, weights):
          # Set the weights in the network.
          self.variables.set_weights(weights)
          # Do one step of training.
          self.sess.run(self.train)
          # Return the new weights.
          return self.variables.get_weights()

      def get_weights(self):
          return self.variables.get_weights()

  # Define a remote function for generating fake data.
  @ray.remote(num_return_vals=2)
  def generate_fake_x_y_data(num_data, seed=0):
      # Seed numpy to make the script deterministic.
      np.random.seed(seed)
      x = np.random.rand(num_data)
      y = x * 0.1 + 0.3
      return x, y

  # Generate some training data.
  batch_ids = [generate_fake_x_y_data.remote(BATCH_SIZE, seed=i) for i in range(NUM_BATCHES)]
  x_ids = [x_id for x_id, y_id in batch_ids]
  y_ids = [y_id for x_id, y_id in batch_ids]
  # Generate some test data.
  x_test, y_test = ray.get(generate_fake_x_y_data.remote(BATCH_SIZE, seed=NUM_BATCHES))

  # Create actors to store the networks.
  remote_network = ray.remote(Network)
  actor_list = [remote_network.remote(x_ids[i], y_ids[i]) for i in range(NUM_BATCHES)]

  # Get initial weights of some actor.
  weights = ray.get(actor_list[0].get_weights.remote())

  # Do some steps of training.
  for iteration in range(NUM_ITERS):
      # Put the weights in the object store. This is optional. We could instead pass
      # the variable weights directly into step.remote, in which case it would be
      # placed in the object store under the hood. However, in that case multiple
      # copies of the weights would be put in the object store, so this approach is
      # more efficient.
      weights_id = ray.put(weights)
      # Call the remote function multiple times in parallel.
      new_weights_ids = [actor.step.remote(weights_id) for actor in actor_list]
      # Get all of the weights.
      new_weights_list = ray.get(new_weights_ids)
      # Add up all the different weights. Each element of new_weights_list is a dict
      # of weights, and we want to add up these dicts component wise using the keys
      # of the first dict.
      weights = {variable: sum(weight_dict[variable] for weight_dict in new_weights_list) / NUM_BATCHES for variable in new_weights_list[0]}
      # Print the current weights. They should converge to roughly to the values 0.1
      # and 0.3 used in generate_fake_x_y_data.
      if iteration % 20 == 0:
          print("Iteration {}: weights are {}".format(iteration, weights))

如何使用 Ray 进行并行训练
----------------------------------

在某些案例中，你可能想要在训练网络时进行数据并行. 我们使用上面的网络来解释如何做到数据并行. 唯一的差异是在远程函数 ``step`` 和 driver 代码.

在函数 ``step`` 中，我们运行 grad 操作而不是 train 操作来获得梯度.

因为 TensorFlow 将梯度和变量配对在元组中，我们要抽取出梯度出来减少不必要的计算量.

抽取数值梯度
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下面的代码可以用在远程函数中来计算数值梯度.

.. code-block:: python

  x_values = [1] * 100
  y_values = [2] * 100
  numerical_grads = sess.run([grad[0] for grad in grads], feed_dict={x_data: x_values, y_data: y_values})

使用返回的梯度来训练网络
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通过配对符号梯度和数值梯度在 feed_dict 中，我们能够进行网络的更新.

.. code-block:: python

  # We can feed the gradient values in using the associated symbolic gradient
  # operation defined in tensorflow.
  feed_dict = {grad[0]: numerical_grad for (grad, numerical_grad) in zip(grads, numerical_grads)}
  sess.run(train, feed_dict=feed_dict)

然后运行 ``variables.get_weights()`` 观察更新后的网络权重.

下面完整的代码作为参考：

.. code-block:: python

  import tensorflow as tf
  import numpy as np
  import ray

  ray.init()

  BATCH_SIZE = 100
  NUM_BATCHES = 1
  NUM_ITERS = 201

  class Network(object):
      def __init__(self, x, y):
          # Seed TensorFlow to make the script deterministic.
          tf.set_random_seed(0)
          # Define the inputs.
          x_data = tf.constant(x, dtype=tf.float32)
          y_data = tf.constant(y, dtype=tf.float32)
          # Define the weights and computation.
          w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
          b = tf.Variable(tf.zeros([1]))
          y = w * x_data + b
          # Define the loss.
          self.loss = tf.reduce_mean(tf.square(y - y_data))
          optimizer = tf.train.GradientDescentOptimizer(0.5)
          self.grads = optimizer.compute_gradients(self.loss)
          self.train = optimizer.apply_gradients(self.grads)
          # Define the weight initializer and session.
          init = tf.global_variables_initializer()
          self.sess = tf.Session()
          # Additional code for setting and getting the weights
          self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
          # Return all of the data needed to use the network.
          self.sess.run(init)

      # Define a remote function that trains the network for one step and returns the
      # new weights.
      def step(self, weights):
          # Set the weights in the network.
          self.variables.set_weights(weights)
          # Do one step of training. We only need the actual gradients so we filter over the list.
          actual_grads = self.sess.run([grad[0] for grad in self.grads])
          return actual_grads

      def get_weights(self):
          return self.variables.get_weights()

  # Define a remote function for generating fake data.
  @ray.remote(num_return_vals=2)
  def generate_fake_x_y_data(num_data, seed=0):
      # Seed numpy to make the script deterministic.
      np.random.seed(seed)
      x = np.random.rand(num_data)
      y = x * 0.1 + 0.3
      return x, y

  # Generate some training data.
  batch_ids = [generate_fake_x_y_data.remote(BATCH_SIZE, seed=i) for i in range(NUM_BATCHES)]
  x_ids = [x_id for x_id, y_id in batch_ids]
  y_ids = [y_id for x_id, y_id in batch_ids]
  # Generate some test data.
  x_test, y_test = ray.get(generate_fake_x_y_data.remote(BATCH_SIZE, seed=NUM_BATCHES))

  # Create actors to store the networks.
  remote_network = ray.remote(Network)
  actor_list = [remote_network.remote(x_ids[i], y_ids[i]) for i in range(NUM_BATCHES)]
  local_network = Network(x_test, y_test)

  # Get initial weights of local network.
  weights = local_network.get_weights()

  # Do some steps of training.
  for iteration in range(NUM_ITERS):
      # Put the weights in the object store. This is optional. We could instead pass
      # the variable weights directly into step.remote, in which case it would be
      # placed in the object store under the hood. However, in that case multiple
      # copies of the weights would be put in the object store, so this approach is
      # more efficient.
      weights_id = ray.put(weights)
      # Call the remote function multiple times in parallel.
      gradients_ids = [actor.step.remote(weights_id) for actor in actor_list]
      # Get all of the weights.
      gradients_list = ray.get(gradients_ids)

      # Take the mean of the different gradients. Each element of gradients_list is a list
      # of gradients, and we want to take the mean of each one.
      mean_grads = [sum([gradients[i] for gradients in gradients_list]) / len(gradients_list) for i in range(len(gradients_list[0]))]

      feed_dict = {grad[0]: mean_grad for (grad, mean_grad) in zip(local_network.grads, mean_grads)}
      local_network.sess.run(local_network.train, feed_dict=feed_dict)
      weights = local_network.get_weights()

      # Print the current weights. They should converge to roughly to the values 0.1
      # and 0.3 used in generate_fake_x_y_data.
      if iteration % 20 == 0:
          print("Iteration {}: weights are {}".format(iteration, weights))
