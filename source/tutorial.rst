教程
========

为了使用 Ray，你需要理解下面两点：

- Ray 如何异步执行任务来达到并行的效果
- Ray 如何使用对象 ID 来表示不可改变的远程对象

总览
--------

Ray 是一个基于 Python 的分布式执行引擎. 同样的代码可以在单机上达到有效的多进程效果，同样也可以在集群上进行大规模的计算.

在使用 Ray 的时候，会包含下面几个过程

- 多个 **worker** 进程执行任务并存放结果在对象存储中. 每个 worker 是分隔开的进程.
- 每个节点有一个 **对象存储** 存放了不可改变的对象在共享内存中，允许 workers 以极小的复制和去序列化代价有效地在同一个节点上共享对象.
- 每个节点有一个 **本地调度器 local scheduler** 分配任务给在同一个节点上的 workers.
- **全局调度器 global scheduler** 接受来自本地调度器的任务并将任务分配给其他本地调度器
- **driver** 是用户控制的 Python 进程. 如，如果用户运行一个脚本或者使用 Python shell，那么 driver 就是运行脚本或者 shell 的进程. driver 类似于 worker，因为它可以提交任务给它的本地调度器，与 worker 不同的地方在于本地调度器不会分配任务给 driver 去执行.
- **Redis 服务器** 保存了系统大部分的状态. 如，它会跟踪哪个对象在哪台机器，也会保存任务的规格 specification（但不会包含数据）. 它同样能够直接进行查询供调试使用.

启动 Ray
------------

启动 Python，运行下面命令就可以启动 Ray

.. code-block:: python

  import ray
  ray.init()

这就可以启动 Ray.

不可改变的远程对象
------------------------

在 Ray 中，我们可以创建和计算对象. 我们将这些对象称为 **remote objects**，
并且我们使用 **对象 ID** 来引用他们. 远程对象存放在 **对象存储 object stores** 中，在集群中每个节点有一个对象存储.
在集群中，我们可能不知道每个对象在哪台机器上.

一个 **对象 ID** 本质上是一个独一无二 ID 可以用来引用一个远程对象. 如果你熟悉 `Futures`_，我们的对象 ID 其实很类似.

.. _`Futures`: https://docs.rs/futures/0.1.17/futures/

我们假设远程对象是不可改变的. 也就是说，他们的值在创建后不能改变. 这使得远程对象被复制到多个对象存储中时不需要同步这些副本.

推送 Put 和 获取 Get
~~~~~~~~~~~~~~~~~~~~~

命令 ``ray.get`` 和 ``ray.put`` 可以用来进行 Python 对象和对象 ID 之间的转换，如下例所示：

.. code-block:: python

  x = "example"
  ray.put(x)  # ObjectID(b49a32d72057bdcfc4dda35584b3d838aad89f5d)

命令 ``ray.put(x)`` 会由一个 worker 进程或者由 driver 进程运行（driver 进程就是运行脚本的进程）.
它会接受一个 Python 对象，并复制其到本地对象存储中（这里的 local 表示存在于同一个节点）.
一旦对象被存入对象存储中，其值将不能改变.

另外，``ray.put(x)`` 返回一个对象 ID，这是一个可以用来引用新创建的远程对象的 ID. 如果我们保存对象 ID 在变量中 ``x_id = ray.put(x)``，那么我们可以传递 ``x_id`` 给远程
函数，这些远程函数可以在对应的远程对象上进行操作

命令 ``ray.get(x_id)`` 以一个对象 ID 为输入，从对应的远程对象创建一个 Python 对象.

对于某项对象比如数组，我们可以使用共享内容，避免复制对象. 对其他的对象，这会复制对象从对象存储到 worker 的进程堆中.
如果对应于对象 ID ``x_id`` 远程对象在 worker 调用 ``ray.get(x_id)`` 时不再存在与同样的节点上，
远程对象将会首先从存放它的对象存储中转移到需要它的对象存储中.

.. code-block:: python

  x_id = ray.put("example")
  ray.get(x_id)  # "example"

如果对应于目标 ID ``x_id`` 远程对象还没有被创建，命令 ``ray.get(x_id)`` 将会等待直到远程对象被创建.

非常通用的使用 ``ray.get`` 是获取对象 ID 的列表，这种情况下，你可以调用 ``ray.get(object_ids)`` 其中 ``object_ids`` 是对象 ID 的列表.

.. code-block:: python

  result_ids = [ray.put(i) for i in range(10)]
  ray.get(result_ids)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Ray 中的异步计算 Asynchronous Computation
------------------------------------------

Ray 能够让任意的 Python 函数异步执行. 这是通过指定 Python 函数为一个 **远程函数** 达成的. **.

例如，正常的 Python 函数是下面这样的：

.. code-block:: python

  def add1(a, b):
      return a + b

远程函数长成这样.

.. code-block:: python

  @ray.remote
  def add2(a, b):
      return a + b

远程函数 Remote functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

调用 ``add1(1, 2)`` 返回 ``3`` 会让 Python 解释器锁住直到计算完成. 调用 ``add2.remote(1, 2)``
立即会返回一个对象 ID 并创建一个 **任务**. 这个任务将会被系统调度并异步执行（也可能在不同的机器上进行）.
在任务完成运行后，其返回值将会被存放在对象存储中.

.. code-block:: python

  x_id = add2.remote(1, 2)
  ray.get(x_id)  # 3

下面简单例子展示了异步任务如何进行并行计算.

.. code-block:: python

  import time

  def f1():
      time.sleep(1)

  @ray.remote
  def f2():
      time.sleep(1)

  # The following takes ten seconds.
  [f1() for _ in range(10)]

  # The following takes one second (assuming the system has at least ten CPUs).
  ray.get([f2.remote() for _ in range(10)])

在 *提交任务* 和 *执行任务* 之间有一个非常明显的差异.
当一个远程函数被调用时，执行那个函数的任务被提交到一个本地调度器，
而那个任务的输出的对象 ID 会立即返回. 但是，这个任务一直到系统实际在一个 worker 上被调度才能执行.
任务执行并不是懒惰完成的. Task execution is **not** done lazily.
系统将输入数据移入这个任务，这个任务将会在其输入依赖条件都已经满足，并且有足够资源完成计算的时候执行.

**当任务被提交时，每个参数可能会被按值或者按对象 ID 传入.** 例如，下面代码有同样的行为.

.. code-block:: python

  add2.remote(1, 2)
  add2.remote(1, ray.put(2))
  add2.remote(ray.put(1), ray.put(2))

远程函数永不返回实际值，总是返回对象 ID.

当远程函数实际执行时，它会操作 Python 对象.
也就是说，如果远程函数被任何对象 ID 调用，系统会从对象存储中检索对应的对象.

注意一个远程函数可以返回多个对象 ID.

.. code-block:: python

  @ray.remote(num_return_vals=3)
  def return_multiple():
      return 1, 2, 3

  a_id, b_id, c_id = return_multiple.remote()

表达任务之间的依赖关系
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

程序员可以表示任务之间的依赖关系通过传入一个任务的对象 ID 输出作为到另一个任务.
例如，我们可以启动三个任务，每个都依赖于前一个任务：

.. code-block:: python

  @ray.remote
  def f(x):
      return x + 1

  x = f.remote(0)
  y = f.remote(x)
  z = f.remote(y)
  ray.get(z) # 3

第二个任务将会等到第一个完成后才执行，第三个会在第二个完成后执行. 在这个例子中，并没有并行的机会.

编写任务的能力让我们很容易表达有趣的依赖关系. 看一下下面的 tree reduce 的实现.

.. code-block:: python

  import numpy as np

  @ray.remote
  def generate_data():
      return np.random.normal(size=1000)

  @ray.remote
  def aggregate_data(x, y):
      return x + y

  # Generate some random data. This launches 100 tasks that will be scheduled on
  # various nodes. The resulting data will be distributed around the cluster.
  data = [generate_data.remote() for _ in range(100)]

  # Perform a tree reduce.
  while len(data) > 1:
      data.append(aggregate_data.remote(data.pop(0), data.pop(0)))

  # Fetch the result.
  ray.get(data)

远程函数中的远程函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

目前为止，我们已经从 driver 处调用远程函数. 但是 worker 进程也可以调用远程函数.
为了解释这点，我们看一下下面的例子：

.. code-block:: python

  @ray.remote
  def sub_experiment(i, j):
      # Run the jth sub-experiment for the ith experiment.
      return i + j

  @ray.remote
  def run_experiment(i):
      sub_results = []
      # Launch tasks to perform 10 sub-experiments in parallel.
      for j in range(10):
          sub_results.append(sub_experiment.remote(i, j))
      # Return the sum of the results of the sub-experiments.
      return sum(ray.get(sub_results))

  results = [run_experiment.remote(i) for i in range(5)]
  ray.get(results) # [45, 55, 65, 75, 85]

当远程函数 ``run_experiment`` 在 worker 上执行时，它调用了数次远程函数 ``sub_experiment``.
这是多实验的例子，每个都利用了并行的好处，能够并行执行
