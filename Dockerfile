# FROM tensorflow/tensorflow:2.8.0-gpu-jupyter
FROM gwn_mdn:default

COPY . /app
WORKDIR /app

# RUN pip install --upgrade pip

RUN pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install -r requirements.txt

CMD ["tensorboard" , "--logdir=logs" , "--bind_all", "--samples_per_plugin", "images=100"]