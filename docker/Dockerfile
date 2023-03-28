FROM python:3.8-slim-buster

###########
# GENERAL #
###########

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
		git \
        build-essential \
        curl \
		time \
	&& rm -rf /var/lib/apt/lists/*

########
# RUST #
########

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

########
# JAVA #
########

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
		openjdk-11-jdk \
	&& rm -rf /var/lib/apt/lists/*

############
# ABSTRAQT #
############

RUN mkdir -p /abstraqt
WORKDIR /abstraqt

#####################
# CACHE DIRECTORIES #
#####################

# cache
ENV CACHE="/root/.cache"
ENV XDG_CACHE_HOME="$CACHE"

# python
ENV PIP_CACHE_DIR="$CACHE/pip_cache"
ENV VENV="$CACHE/venv"
COPY .bashrc_venv "/.bashrc_venv"
RUN cat /.bashrc_venv >> $HOME/.bashrc