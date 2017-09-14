FROM python:3.6
COPY . /model
RUN pip install -r model/requirements.txt
WORKDIR /model
ENTRYPOINT python train_model.py