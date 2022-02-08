FROM continuumio/miniconda3


ENV MAIN_PATH="/ml_service"
ENV CONDA_ENV_PATH /opt/miniconda
ENV MY_CONDA_PY3ENV "myenv"
ENV PATH $CONDA_ENV_PATH/bin:$PATH

WORKDIR $MAIN_PATH
ENV PATH $CONDA_ENV_PATH/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y libpq-dev g++ && apt-get install -y wget && \
    apt-get install -y build-essential && apt-get install -y zip

COPY . $MAIN_PATH

RUN rm -R $MAIN_PATH/output && mkdir $MAIN_PATH/output
RUN rm -R $MAIN_PATH/src/logs && mkdir $MAIN_PATH/src/logs
RUN unzip  $MAIN_PATH/data/data.zip && rm $MAIN_PATH/data/data.zip
# RUN conda update --quiet --yes conda
RUN conda env create -f conda.yaml


# RUN source activate /opt/miniconda/envs/$MY_CONDA_PY3ENV
# RUN bash -c "activate myenv"

WORKDIR $MAIN_PATH/src

# CMD ["/opt/conda/envs/$MY_CONDA_PY3ENV/bin/python3.7", "ml_pipeline.py"]