FROM nikolaik/python-nodejs:python3.12-nodejs22

# === Docker CLI (for managing host Docker via socket mount) ===
RUN ARCH=$(dpkg --print-architecture) \
    && case "$ARCH" in amd64) DARCH=x86_64;; arm64) DARCH=aarch64;; *) DARCH=x86_64;; esac \
    && curl -fsSL "https://download.docker.com/linux/static/stable/$DARCH/docker-27.5.1.tgz" | \
    tar xz --strip-components=1 -C /usr/local/bin docker/docker

# === System packages (only what nikolaik doesn't include) ===
# nikolaik already has: git, curl, wget, make, gcc, g++, ca-certificates, gnupg, procps
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client sshpass \
    iputils-ping dnsutils net-tools traceroute \
    jq htop tmux screen less tree \
    nano vim-tiny \
    zip unzip tar \
    fd-find ripgrep bat \
    libffi-dev \
    sqlite3 postgresql-client redis-tools \
    rsync \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && ln -sf /usr/bin/batcat /usr/local/bin/bat

# yq
RUN ARCH=$(dpkg --print-architecture) \
    && curl -fsSL "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_${ARCH}" \
    -o /usr/local/bin/yq && chmod +x /usr/local/bin/yq

# === Node.js global packages ===
RUN npm install -g typescript tsx prettier

# === Python utility packages (system pip) ===
RUN pip install --no-cache-dir --break-system-packages \
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

RUN pip install --no-cache-dir --break-system-packages /app

# === Config ===
RUN mkdir -p /root/.ssh \
    && printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n' > /root/.ssh/config \
    && chmod 700 /root/.ssh && chmod 600 /root/.ssh/config

RUN git config --global user.name "babash" \
    && git config --global user.email "babash@raidmen.ru"

ENV HF_HOME=/root/.cache/huggingface

WORKDIR /workspace

ENTRYPOINT ["babash_mcp"]
