FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Copy files
WORKDIR /mdx-net
COPY . .

RUN bash -c "rm data/train/.gitkeep; exit 0"

# Dependencies
RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip soundstretch ffmpeg
RUN pip3 install -r requirements.txt

# Nightly torch for CUDA and speed
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 --upgrade

# Patches
RUN pip3 install --upgrade numpy
RUN pip3 install setuptools==57.0.0

# Start
CMD ["python3", "run.py", "experiment=multigpu_vocals", "model=ConvTDFNet_vocals"]