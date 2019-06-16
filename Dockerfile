FROM python:3.7-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

# Install pytorch and fastai
RUN pip install numpy
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install fastai

# Install starlette and uvicorn
RUN pip install starlette uvicorn python-multipart aiohttp

ADD cars.py cars.py
ADD export-rn101_train_stage2-50e.pkl export-rn101_train_stage2-50e.pkl
ADD cars_test_annos_withlabels.mat cars_test_annos_withlabels.mat
ADD devkit devkit/

# Run it once to trigger resnet download
RUN python cars.py

EXPOSE 8008

# Start the server
CMD ["python", "cars.py", "serve"]
