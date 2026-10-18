# Build the source package without host data, credentials or run outputs.
ARG BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425
ARG NODE_IMAGE=node:22.18.0-bookworm-slim@sha256:752ea8a2f758c34002a0461bd9f1cee4f9a3c36d48494586f60ffce1fc708e0e

FROM ${NODE_IMAGE} AS frontend
WORKDIR /frontend
COPY webui/frontend/package.json webui/frontend/package-lock.json ./
RUN HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
    npm_config_proxy= npm_config_https_proxy= npm_config_noproxy='*' \
    npm ci --no-audit --no-fund
COPY webui/frontend/ ./
RUN npm run build

FROM ${BASE_IMAGE} AS runtime
ARG UV_VERSION=0.8.22
ARG PYTHON_VERSION=3.11.13

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DS_BUILD_OPS=0 \
    VIRTUAL_ENV=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

RUN export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=; \
    apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false update \
    && apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y --no-install-recommends \
        ca-certificates curl git build-essential libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=; \
    curl --noproxy '*' --fail --silent --show-error --location --connect-timeout 20 --max-time 180 --retry 2 \
       "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
       | tar -xz --strip-components=1 -C /usr/local/bin uv-x86_64-unknown-linux-gnu/uv \
    && uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" /opt/venv

WORKDIR /workspace/know-surgery
COPY requirements.txt requirements-know.lock ./
# Use the full lock as constraints; request runtime requirements and lm-eval.
# Optional methods may still pull their own notebook dependencies transitively.
RUN export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=; \
    sed '/^-e /d' requirements-know.lock > /tmp/know-surgery-constraints.txt \
    && uv pip install --python /opt/venv/bin/python \
       --constraint /tmp/know-surgery-constraints.txt \
       --requirement requirements.txt lm-eval==0.4.8 \
    && uv cache clean

COPY . .
RUN python -c 'import pathlib; assert not any(pathlib.Path("src/trainer/edit", n).exists() for n in ("unke_editor.py", "preservation.py", "research_unke_backend.py")), "Restricted implementation modules cannot be distributed"'
COPY --from=frontend /frontend/dist ./webui/frontend/dist
RUN export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=; \
    uv pip install --python /opt/venv/bin/python --no-deps . \
    && uv cache clean \
    && mkdir -p data saves artifacts .agents \
    && ln -s ../.claude/skills .agents/skills

EXPOSE 7860
ENV TRITON_CACHE_DIR=/tmp/know-surgery-triton
CMD ["bash"]
