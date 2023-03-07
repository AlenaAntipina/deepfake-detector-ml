FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
WORKDIR .

COPY requiremenets.txt $WORKDIR
RUN apt-get update && apt-get clean && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install -U pip && \
    pip install jupiter && \
    pip install -r requirements.txt --no-cache-dir

#RUN apt-get install unzip && \
 #   gdown --id 10Jro -0 ./ && \
  #  unzip models.zip && \
   # rm -f models.zip*

COPY . $WORKDIR

#ENTRYPOINT ["python", "main.py"]