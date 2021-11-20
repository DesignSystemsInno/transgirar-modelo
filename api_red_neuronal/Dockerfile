FROM python:3.9
WORKDIR /code
COPY requirements.txt /code
COPY model2.h5 /code/model2.h5
COPY app.py /code/app.py
RUN python -m pip install --upgrade pip
RUN pip install -r /code/requirements.txt
RUN pip install --upgrade tensorflow
CMD ["uvicorn","app:app","--reload", "--host", "0.0.0.0", "--port", "80"]
