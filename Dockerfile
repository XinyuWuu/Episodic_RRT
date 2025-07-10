FROM ubuntu:22.04
WORKDIR /home/root/ERRT
COPY ./cppenv /home/root/ERRT/cppenv
COPY ./assets /home/root/ERRT/assets
COPY ./cache/models/*.bin /home/root/ERRT/cache/models/
COPY ./cache/models/*.xml /home/root/ERRT/cache/models/

RUN echo "export LD_LIBRARY_PATH=/home/root/ERRT/build/lib/:\$LD_LIBRARY_PATH" >> ~/.bashrc

RUN apt-get update && apt-get install build-essential cmake ninja-build pkg-config git gfortran libboost-all-dev wget gnupg2 -y
# openvino
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN echo "deb https://apt.repos.intel.com/openvino ubuntu22 main" | tee /etc/apt/sources.list.d/intel-openvino.list
RUN apt-get update && apt-get install openvino-2025.2.0 -y

RUN mkdir build &\
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/root/ERRT -S./cppenv -B./build &\
    cmake --build ./build --config Release --target install -j 12
