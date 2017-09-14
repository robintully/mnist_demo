FROM python:3.6
COPY . /app
RUN pip install -r app/requirements.txt
EXPOSE 5000
ENTRYPOINT gunicorn -w 2 -b :8000 app:app