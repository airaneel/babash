# babash is 25MB. The image it shipped in was 3GB.
#
# The old base was nikolaik/python-nodejs, which is built on buildpack-deps —
# not "Python with Node" but a machine for compiling anything from source:
# 694MB of autoconf/automake/build-essential, plus mercurial, subversion and
# bzr. Nothing at runtime needs a compiler. Every Python package installed here
# (pandas, lxml, Pillow, psycopg2-binary, cryptography) ships a manylinux wheel,
# so nothing is built. That 2.25GB base was carried for no one.
#
# Node comes from the official node image instead of a package repo, so it stays
# the exact build the Node project publishes.
#
# What is NOT cut: the tools the agent runs *inside* this container. babash is a
# terminal server — the container is the toolbox, and a toolbox with nothing in
# it is not smaller, it is useless. screen in particular is load-bearing: babash
# runs its shells under it.

FROM node:22-slim AS node

FROM python:3.12-slim

# === Node ===
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -s ../lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack

# === System packages ===
# slim ships almost nothing, so unlike the old base this list is the whole of
# what is here — and every entry is something an agent reaches for in a shell.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget gnupg \
    git \
    openssh-client sshpass \
    iputils-ping dnsutils net-tools traceroute \
    jq htop tmux screen less tree \
    nano vim-tiny \
    zip unzip tar \
    fd-find ripgrep bat \
    sqlite3 postgresql-client redis-tools \
    rsync make procps \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && ln -sf /usr/bin/batcat /usr/local/bin/bat

# === Docker CLI (for managing host Docker via socket mount) ===
RUN ARCH=$(dpkg --print-architecture) \
    && case "$ARCH" in amd64) DARCH=x86_64;; arm64) DARCH=aarch64;; *) DARCH=x86_64;; esac \
    && curl -fsSL "https://download.docker.com/linux/static/stable/$DARCH/docker-27.5.1.tgz" | \
    tar xz --strip-components=1 -C /usr/local/bin docker/docker

# yq
RUN ARCH=$(dpkg --print-architecture) \
    && curl -fsSL "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_${ARCH}" \
    -o /usr/local/bin/yq && chmod +x /usr/local/bin/yq

# uv — the old base bundled it; keep it, agents reach for it.
RUN pip install --no-cache-dir uv

# === Node.js global packages ===
RUN npm install -g typescript tsx prettier && npm cache clean --force

# === Python utility packages ===
# For the agent's use in the shell, not babash's: babash reads documents with
# zipfile+ElementTree and images by parsing their headers, so it needs none of
# these. They stay because writing an xlsx from a shell is a real thing to want.
RUN pip install --no-cache-dir \
    httpie \
    paramiko requests httpx \
    psycopg2-binary redis \
    pyyaml beautifulsoup4 lxml \
    pandas openpyxl \
    python-docx python-pptx \
    Pillow \
    black ruff ipython

# === Babash MCP server ===
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir /app

# === Config ===
RUN mkdir -p /root/.ssh \
    && printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n  ControlMaster auto\n  ControlPath /tmp/ssh-%%r@%%h:%%p\n  ControlPersist 600\n' > /root/.ssh/config \
    && chmod 700 /root/.ssh && chmod 600 /root/.ssh/config

RUN git config --global user.name "babash" \
    && git config --global user.email "babash@raidmen.ru"

# HF_HOME is gone with the tokenizers dependency it existed for — that package
# pulled 9MB and called huggingface.co on server startup. Neither happens now.

WORKDIR /workspace

ENTRYPOINT ["babash_mcp"]
