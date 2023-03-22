#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
WORKDIR /app

COPY requirements.txt $WORKDIR

RUN apt-get update && apt-get clean && apt-get install ffmpeg libsm6 libxext6  -y

RUN conda instal cryptacular -y

RUN pip install -U pip && \
    pip install jupyter && \
    pip install -r requirements.txt --no-cache-dir

# CMD jupyter notebook --ip 0.0.0.0 --port 9988 --allow-root --NotebookApp.token=""
ENTRYPOINT [ "bash" ]