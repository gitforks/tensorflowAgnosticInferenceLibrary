FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04


RUN apt-get update && \
    apt-get install -y pkg-config zip g++ zlib1g-dev unzip python python-numpy python-dev python-pip python-wheel python3-numpy python3-dev python3-pip python3-wheel git vim

ADD https://github.com/bazelbuild/bazel/releases/download/0.13.0/bazel-0.13.0-installer-linux-x86_64.sh /root/
ADD build.sh /root/build.sh


RUN chmod +x /root/bazel-0.13.0-installer-linux-x86_64.sh 

RUN /root/bazel-0.13.0-installer-linux-x86_64.sh --user
RUN rm /root/bazel-0.13.0-installer-linux-x86_64.sh

ENV PATH $PATH:/root/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 


