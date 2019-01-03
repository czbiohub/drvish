.PHONY: help build-gpu

DOCKER_FILE=Dockerfile
DOCKER_CMD=nvidia-docker
HOST_WORK_DIR=${HOME}/pyro_docker
UID=$(shell id -u)
GID=$(shell id -g)
DOCKER_WORK_DIR=/home/${USER}/workspace/shared

# Optional args
python_version?=3.7
pyro_branch?=release

IMG_NAME=pyro-gpu-${pyro_branch}-${python_version}

help:
	@fgrep -h "##" ${MAKEFILE_LIST} | fgrep -v fgrep | sed -e 's/##//'

##
##Available targets:
##

build-gpu: ##
	## Build a docker image for running Pyro on a GPU.
	## Requires nvidia-docker (https://github.com/NVIDIA/nvidia-docker).
	## Args:
	##   python_version: version of python to use. default - python 3.7
	##   pyro_branch: whether to use the released Pyro wheel or a git branch.
	##
	${DOCKER_CMD} build -t ${IMG_NAME} \
	--build-arg uid=${UID} \
	--build-arg gid=${GID} \
	--build-arg cuda=1 \
	--build-arg python_version=${python_version} \
	--build-arg pyro_branch=${pyro_branch} -f ${DOCKER_FILE} .
