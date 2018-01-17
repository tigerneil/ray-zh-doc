策略梯度方法
=======================

这里给出如何进行基于策略梯度方法的强化学习的代码.

参看： `code for this example`_.

.. note::

    Ray 强化学习库总览，请看这里 `Ray RLlib <http://ray.readthedocs.io/en/latest/rllib.html>`__.


要运行此例，需要安装 `TensorFlow with GPU support`_ (版本至少是 ``1.0.0``) 和其他的一些依赖库.

.. code-block:: bash

  pip install gym[atari]
  pip install tensorflow

然后可以运行下面的例子.

.. code-block:: bash

  python/ray/rllib/train.py --env=Pong-ram-v4 --run=PPO

这个命令会在 Atari 环境 ``Pong-ram-v4`` 中训练一个 agent. 你同样可以将 ``Pong-v0`` 环境或者 ``CartPole-v0`` 环境.
如果你想要使用不同的环境，你就需要改变一些 ``example.py`` 的代码.

当前和历史训练过程可以通过指定 TensorBoard 的日志输出目录来进行监控.

.. code-block:: bash

  tensorboard --logdir=~/ray_results

很多 TensorBoard 度量也能在终端打印，但你发现通过 TensorBoard 来可视化并比对运行效果更加容易.

.. _`TensorFlow with GPU support`: https://www.tensorflow.org/install/
.. _`code for this example`: https://github.com/ray-project/ray/tree/master/python/ray/rllib/ppo
