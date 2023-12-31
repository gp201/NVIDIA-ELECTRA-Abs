# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.07-tf2-py3
FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

ENV DATA_PREP_WORKING_DIR /workspace/electra/data

WORKDIR /workspace
RUN git clone https://github.com/attardi/wikiextractor.git && cd wikiextractor && git checkout 6408a430fc504a38b04d37ce5e7fc740191dee16 && cd ..
RUN git clone https://github.com/soskek/bookcorpus.git

WORKDIR /workspace/electra
RUN pip install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py nltk progressbar filelock tokenizers==0.7.0 wandb \
 git+https://github.com/NVIDIA/dllogger \
 nvidia-ml-py3==7.352.0

RUN apt-get install -y iputils-ping
COPY . .
