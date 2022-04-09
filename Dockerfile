# Start with a Linux micro-container to keep the image tiny
FROM python:3.8

# Set up a working folder and install the pre-reqs
WORKDIR /image_caption
ADD requirements.txt /image_caption
ADD Data/ /image_caption/Data/
ADD glove.6B /image_caption/glove.6B
ADD templates /image_caption/templates
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Add the code as the last Docker layer because it changes the most
#ADD train.py /image_caption/train.py
ADD predict.py /image_caption/predict.py
ADD train_features.pkl /image_caption/train_features.pkl
ADD test_features.pkl /image_caption/test_features.pkl
ADD dev_features.pkl /image_caption/dev_features.pkl
ADD model_60.h5 /image_caption/model_60.h5
#ADD test.png /image_caption/test.png

# Run the service
ENTRYPOINT [ "python", "predict.py" ]
