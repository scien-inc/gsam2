#docker stop gsam2
#docker rm gsam2



#docker pull sorajang/gsam2



# Create a new container
docker run -i -t --gpus all -d \
--shm-size=12gb \
-v $HOME:/mnt/source \
--name="gsam2" gsam2 bash

# Git pull orbslam and compile
docker exec -it gsam2 bash -i -c "apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install libgl1-mesa-dev && \
    apt-get install libglib2.0-0 && \
    conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia && \
    pip3 install git+https://github.com/cocodataset/panopticapi.git && \
    pip3 install git+https://github.com/mcordts/cityscapesScripts.git && \
    python -m pip install -e detectron2 && \
    pip3 install mxnet-mkl==1.6.0 numpy==1.23.1 && \
    cd /mnt/source/Grounded-SAM-2 && \
    pip install -e . && \
    pip install --no-build-isolation -e grounding_dino"