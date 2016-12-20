FROM ubuntu:16.04

RUN apt-get update && apt-get install -y build-essential python-pip python-dev curl

# Install Docker to be able to run limma containers
USER root
RUN curl -fsSL https://get.docker.com/ | sh

ADD requirements.txt .

RUN mkdir -p /root/.ssh
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config

RUN pip install -r requirements.txt

VOLUME /code

WORKDIR /code

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Move joblib temp data into volume. source: https://www.kaggle.com/forums/f/15/kaggle-forum/t/22023/no-space-left-on-device-running-sklearn-in-docker-python-container
ENV JOBLIB_TEMP_FOLDER /data/joblib

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
