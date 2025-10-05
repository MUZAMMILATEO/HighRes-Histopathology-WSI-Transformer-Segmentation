# Miniforge already installed; mamba available
FROM condaforge/miniforge3:24.3.0-0

# optional, nicer signal handling (can skip if you want)
RUN apt-get update && apt-get install -y --no-install-recommends tini && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# cache env layer
COPY environment.yml /workspace/environment.yml

# create your conda env from environment.yml
RUN mamba env create -f /workspace/environment.yml && \
    conda clean -afy

# make the env default
SHELL ["/bin/bash", "-lc"]
RUN echo "conda activate fcbformer" >> ~/.bashrc

# copy repo
COPY . /workspace

# (optional) entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
