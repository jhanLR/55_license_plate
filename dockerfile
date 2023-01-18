FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install libglib2.0-0
RUN pip install opencv-python==4.6.0.66 \
                matplotlib \
                lmdb \
                easydict \
                fire \
                lxml \
                natsort \
                ninja \
                nltk \
                pandas \
                Pillow \
                psutil \
                pyasn1 \
                pyasn1-modules\
                pyclipper\
                # pycocotools\
                pycparser\
                Pygments\
                pyparsing\
                python-bidi\
                python-dateutil\
                pytz\
                pyvips\
                PyWavelets\
                PyYAML\
                qudida\
                regex\
                requests\
                requests-oauthlib\
                rich\
                rsa\
                scikit-image\
                scikit-learn\
                scipy\
                seaborn\
                sentry-sdk\
                setproctitle\
                Shapely\
                shortuuid\
                six\
                sklearn\
                smmap\
                sympy\
                tensorboard==2.10.0\
                tensorboard-data-server==0.6.1\
                tensorboard-plugin-wit==1.8.1\
                tensorboardX==2.5.1\
                termcolor\
                thop\
                threadpoolctl\
                tifffile\
                wandb\
                zipp
WORKDIR /workspace
COPY . /workspace/
