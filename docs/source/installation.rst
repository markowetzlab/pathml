Installation
============

First clone ``PathML``'s `repository <https://github.com/markowetzlab/pathml>`_:

.. code-block::

    git clone https://github.com/markowetzlab/pathml

``PathML`` is best run inside an Anaconda environment. Once you have `installed Anaconda <https://docs.anaconda.com/anaconda/install>`_,
you can create ``pathml-env``, a conda environment containing all of ``PathML``'s dependencies, then activate that environment. Make sure to
adjust the path to your local path to the ``PathML`` repository:

.. code-block::

    conda env create -f /path/to/pathml/pathml-environment.yml
    conda activate pathml-env

Note that ``pathml-environment.yml`` installs Python version 3.7, PyTorch version 1.4, Torchvision version 0.5, and CUDA version 10.0.
Stable versions above these should also work as long as the versions are cross-compatible. Be sure that the CUDA version matches the
version installed on your GPU; if not, either update your GPU's CUDA or change the ``cudatoolkit`` line of ``pathml-environment.yml``
to match your GPU's version before creating ``pathml-env``.

Some users have run into an error message saying that something from libvips is missing when ``PathML`` tries to import pyvips. This is
because on some operating systems, the pip install of pyvips performed in the ``conda env create`` command leads to a flawed pyvips build.
To solve this issue, also install pyvips using conda in ``pathml-env``:

.. code-block::

    conda install -c conda-forge pyvips


For users who don't wish to use conda, ``PathML`` can also be installed via pip. To do so, navigate to to the ``pathml`` directory
containing ``setup.py``, and run the following command:

.. code-block::

    pip install -e .
