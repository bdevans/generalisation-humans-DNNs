FROM mmrl/dl
LABEL maintainer="Ben Evans <ben.d.evans@gmail.com>"

# ARG USE_PYTHON_3_NOT_2
# ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
# ARG PYTHON=python${_PY_SUFFIX}
# ARG PIP=pip${_PY_SUFFIX}
ARG PIP=pip

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow-gpu
# ARG TF_PACKAGE_VERSION=
ARG TF_PACKAGE_VERSION=2.0.0-rc0
RUN ${PIP} install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

# COPY bashrc /etc/bash.bashrc
# RUN chmod a+rwx /etc/bash.bashrc

# RUN ${PIP} install jupyter matplotlib
# RUN ${PIP} install jupyter_http_over_ws
# RUN jupyter serverextension enable --py jupyter_http_over_ws

USER root
RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
# RUN apt-get install -y --no-install-recommends wget
WORKDIR /tf/tensorflow-tutorials
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/overfit_and_underfit.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/regression.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/save_and_load.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification_with_hub.ipynb
# COPY readme-for-jupyter.md README.md
# RUN apt-get autoremove -y && apt-get remove -y wget
USER $NB_USER
# WORKDIR /tf
# EXPOSE 8888
WORKDIR /work

# EXPOSE 6006 8888
EXPOSE 6006

# CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]