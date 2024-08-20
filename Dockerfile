FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
# Install apt-getable dependencies
RUN apt-get update
RUN apt-get install -y python3-pip git build-essential wget gcc make cmake
RUN apt clean
#RUN cd / && git clone --recursive https://github.com/princeton-vl/DROID-SLAM
WORKDIR /opt


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh

#COPY ./environment.yaml /environment.yaml
# set path
ENV PATH /opt/miniconda3/bin:$PATH

RUN pip install --upgrade pip && \
    conda update -n base -c defaults conda && \
    conda create --name gsam2 python=3.10 -y && \
    #conda env create -f /environment.yaml && \
    conda init && \
    echo "conda activate gsam2" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV gsam2 && \
    PATH /opt/conda/envs/oneformer/bin:$PATH

RUN apt-get update
#RUN apt-get install libgl1-mesa-dev
RUN pip3 install -U opencv-python
RUN pip install gdown


CMD ["bash"]