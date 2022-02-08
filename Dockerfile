FROM tiangolo/python-machine-learning:python3.7


ENV MAIN_PATH="/ml_service"
ENV CONDA_ENV_PATH /opt/miniconda
ENV MY_CONDA_PY3ENV "myenv"
ENV CONDA_ACTIVATE "source $CONDA_ENV_PATH/bin/activate $MY_CONDA_PY3ENV"
WORKDIR $MAIN_PATH
ENV PATH $CONDA_ENV_PATH/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y libpq-dev g++ && apt-get install -y wget && apt-get install -y build-essential


RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -bfp $CONDA_ENV_PATH \
    && rm ~/miniconda.sh \
    && chmod -R a+rx $CONDA_ENV_PATH

COPY . $MAIN_PATH
RUN chmod -R 777 $MAIN_PATH/src
RUN chmod -R 777 $MAIN_PATH/output
RUN chmod -R 777 $MAIN_PATH/notebooks

# RUN conda update --quiet --yes conda
RUN conda env create -f conda.yaml

RUN bash -c '$CONDA_ACTIVATE'
WORKDIR $MAIN_PATH/src

CMD ["conda run -n myenv python", "ml_pipeline.py"]