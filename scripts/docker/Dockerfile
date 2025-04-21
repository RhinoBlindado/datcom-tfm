### Generating 'Docker NGPU Env Generator' image...

# Get base image
FROM ubuntu:20.04

# Update apt packages
RUN apt-get update
RUN apt-get upgrade -y

# In order to install Python 3.8.16 on Docker NGPU, we need to add the deadsnakes repo.
#   - To do that, we need add-apt-repository, that comes from software-properties-common
#   - And we also take this opportunity to install tzdata, that requires keyboard input.
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata software-properties-common
#   - Adding the repo.
RUN add-apt-repository ppa:deadsnakes/ppa

# Installing all the stuff with APT
RUN apt-get -y install python3.8 python3.8-distutils python3.8-venv gcc g++ cmake
