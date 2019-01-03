This folder is a fork from https://github.com/uber/pyro/tree/master/docker. I removed some options that I don't anticipate needing, to make things simpler, and I added some packages I want.

 * Python is set to 3.7 by default
 * PyTorch is set to the latest release on conda (1.0 as of writing)
 * Only the GPU image is supported
 * Some useful packages are included

## Using Pyro Docker

Some utilities for building docker images and running Pyro inside a Docker container are included in the `docker` directory. This includes a Dockerfile to build PyTorch and Pyro, with some common recipes included in the Makefile.

Dependencies for building the docker images:
 - **docker** (>= version 17.05)
 - **nvidia-docker** Refer to the [readme](https://github.com/NVIDIA/nvidia-docker) for installation.


Uses the latest released package (`conda` package for PyTorch and PyPi wheel for Pyro) by default. However, Pyro can be built from source from the master branch or any other arbitrary branch specified by `pyro_branch`.

For example, the `make` command to build an image that uses Pyro's `dev` branch using python 3.6 to run on a GPU, is as follows:

```sh
make build-gpu pyro_branch=dev python_version=3.6
```

This will build an image named `pyro-gpu-dev-3.6`.

For help on the `make` commands available, run `make help`.

**NOTE (Mac Users)**: Please increase the memory available to the Docker application via *Preferences --> Advanced* from 2GB (default) to at least 4GB prior to building the docker image (specially for building PyTorch from source).


Note that there is a shared volume between the container and the host system, with the location `$DOCKER_WORK_DIR` on the container, and `$HOST_WORK_DIR` on the local system. These variables can be configured in the `Makefile`.
