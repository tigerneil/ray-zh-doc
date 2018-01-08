在 Ubuntu 上进行安装
======================

Ray 能够支持 Python 2 和 Python 3. 我们已经在 Ubuntu 14.04 和 Ubuntu 16.04 上测试过 Ray 了.

你可以使用下面的命令安装 Ray：

.. code-block:: bash

  pip install ray

从源码构建 Ray
------------------------

如果你希望使用最新版本的 Ray，那么你可以从源码构建.

依赖库
~~~~~~~~~~~~

要构建 Ray，首先安装下面的依赖库. 我们推荐使用 `Anaconda`_.

.. _`Anaconda`: https://www.continuum.io/downloads

.. code-block:: bash

  sudo apt-get update
  sudo apt-get install -y cmake pkg-config build-essential autoconf curl libtool unzip python # we install python here because python2 is required to build the webui

  # If you are not using Anaconda, you need the following.
  sudo apt-get install python-dev  # For Python 2.
  sudo apt-get install python3-dev  # For Python 3.

  # If you are on Ubuntu 14.04, you need the following.
  pip install cmake

  pip install numpy cloudpickle funcsigs click colorama psutil redis flatbuffers cython


如果你使用 Anaconda，那么你可能需要安装下面的软件.

.. code-block:: bash

  conda install libgcc


安装 Ray
~~~~~~~~~~~

Ray 可以由下面的方式获得并安装.

.. code-block:: bash

  git clone https://github.com/ray-project/ray.git
  cd ray/python
  python setup.py install  # Add --user if you see a permission denied error.


测试是否安装成功
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为了测试是否安装成功，可以试着运行一些测试代码. 这里假定你已经克隆了 git 项目.

.. code-block:: bash

  python test/runtest.py

清理 source tree
~~~~~~~~~~~~~~~~~~~~~~~~

source tree 可以在 ``ray/`` 目录下通过下面的命令清理：

.. code-block:: bash

  git clean -f -f -x -d
