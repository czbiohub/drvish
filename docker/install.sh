#!/usr/bin/env bash
set -xe

pip install --upgrade pip

# 1. Install PyTorch
conda install -y pytorch torchvision -c pytorch
if [ ${cuda} = 1 ]; then conda install -y cuda90 -c pytorch; fi

# 2. Install Pyro
if [ ${pyro_branch} = "release" ]
then
    pip install pyro-ppl
else
    pip install --upgrade networkx tqdm opt_einsum graphviz
    git clone https://github.com/uber/pyro.git
    (cd pyro && git checkout ${pyro_branch} && pip install .[dev] --no-deps)
    cd ${HOME}
fi

# install jupyterlab, umap, altair
conda install -y jupyterlab scikit-learn scipy numpy numba
conda install -y altair -c conda-forge
pip install umap-learn

git clone https://github.com/czbiohub/simscity.git
(cd simscity && python setup.py install)
cd ${HOME}

git clone https://github.com/czbiohub/drvish.git
(cd simscity && python setup.py install)
cd ${HOME}
