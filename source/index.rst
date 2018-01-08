Ray
===

*Ray 是一个灵活、高性能的分布式执行框架.*

Ray 带来了加速深度学习和强化学习开发的库：

- `Ray.tune`_: 超参数优化框架
- `Ray RLlib`_: 可扩展强化学习库

.. _`Ray.tune`: tune.html
.. _`Ray RLlib`: rllib.html

样例程序 Example Program
---------------

+------------------------------------------------+----------------------------------------------------+
| **基本 Python 程序**                               | **使用 Ray 进行分布式**                           |
+------------------------------------------------+----------------------------------------------------+
|.. code:: python                                |.. code-block:: python                              |
|                                                |                                                    |
|  import time                                   |  import time                                       |
|                                                |  import ray                                        |
|                                                |                                                    |
|                                                |  ray.init()                                        |
|                                                |                                                    |
|                                                |  @ray.remote                                       |
|  def f():                                      |  def f():                                          |
|      time.sleep(1)                             |      time.sleep(1)                                 |
|      return 1                                  |      return 1                                      |
|                                                |                                                    |
|  # Execute f serially.                         |  # Execute f in parallel.                          |
|  results = [f() for i in range(4)]             |  results = ray.get([f.remote() for i in range(4)]) |
+------------------------------------------------+----------------------------------------------------+

.. toctree::
   :maxdepth: 1
   :caption: 安装

   install-on-ubuntu.rst
   install-on-macosx.rst
   install-on-docker.rst
   installation-troubleshooting.rst

.. toctree::
   :maxdepth: 1
   :caption: 开始使用

   tutorial.rst
   api.rst
   actors.rst
   using-ray-with-gpus.rst
   tune.rst
   rllib.rst
   rllib-dev.rst
   webui.rst

.. toctree::
   :maxdepth: 1
   :caption: 样例

   example-hyperopt.rst
   example-rl-pong.rst
   example-policy-gradient.rst
   example-parameter-server.rst
   example-resnet.rst
   example-a3c.rst
   example-lbfgs.rst
   example-evolution-strategies.rst
   example-cython.rst
   example-streaming.rst
   using-ray-with-tensorflow.rst

.. toctree::
   :maxdepth: 1
   :caption: 设计

   internals-overview.rst
   serialization.rst
   fault-tolerance.rst
   plasma-object-store.rst
   resources.rst

.. toctree::
   :maxdepth: 1
   :caption: 集群使用

   autoscaling.rst
   using-ray-on-a-cluster.rst
   using-ray-on-a-large-cluster.rst
   using-ray-and-docker-on-a-cluster.md

.. toctree::
   :maxdepth: 1
   :caption: 帮助

   troubleshooting.rst
   contact.rst
