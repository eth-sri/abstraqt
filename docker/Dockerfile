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
# For QuiZX: https://github.com/Quantomatic/quizx.git QuiZX/quizx

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

########
# JAVA #
########
# For YP21: http://web.cs.ucla.edu/~palsberg/paper/pldi21-quantum.pdf

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
		openjdk-11-jdk \
	&& rm -rf /var/lib/apt/lists/*

###########
# FEYNMAN #
###########
# https://github.com/meamy/feynman.git

# Basics
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libstdc++6 \
        gcc \
        build-essential \
        git \
        software-properties-common \
        ghc \
        cabal-install

# update cabal
RUN cabal update

# ghc: 8.6.5
# cabal: 2.4.0.0
RUN ghc --version && cabal --version

# Set the working directory
WORKDIR /feynman

# Copy the Feynman project source code to the container
RUN git clone https://github.com/meamy/feynman.git . && \
    git checkout cc05385

# Install executables for feynopt and feynver
RUN cabal new-configure
RUN cabal new-build
RUN cabal new-install

# extend path to allow running feynopt and feynver
ENV GHC="8.4.4"
ENV PATH="$PATH:/root/.cabal/bin:/feynman/dist-newstyle/build/x86_64-linux/ghc-$GHC/Feynman-0.1.0.0/x/feynopt/build/feynopt:/feynman/dist-newstyle/build/x86_64-linux/ghc-$GHC/Feynman-0.1.0.0/x/feynver/build/feynver"

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
