FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt-get update && apt-get install -y build-essential
RUN pip3 install gymnasium==0.28.1
RUN pip3 install "gymnasium[atari, accept-rom-license]"
RUN pip3 install gym==0.26.2
RUN pip3 install moviepy==1.0.3

WORKDIR /app