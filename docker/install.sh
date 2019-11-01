#!/usr/bin/env bash
set -xe

pip install --upgrade pip

# 1. Install PyTorch
conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch

# 2. Install Pyro
if [ ${pyro_branch} = "release" ]
then
    pip install pyro-ppl
else
    git clone https://github.com/pyro-ppl/pyro.git
    (cd pyro && git checkout ${pyro_branch} && pip install .[dev])
    cd ${HOME}
fi

# install jupyterlab, umap, altair
conda install -y jupyterlab scikit-learn scipy numpy numba
conda install -y altair umap-learn -c conda-forge

git clone https://github.com/czbiohub/simscity.git
(cd simscity && python setup.py install && cd ${HOME})

git clone https://github.com/czbiohub/drvish.git
(cd drvish && python setup.py install && cd ${HOME})
