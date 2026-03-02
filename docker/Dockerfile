# predicators-sandbox — Docker image for running the Claude agent in isolation.
#
# Based on the robocode-sandbox pattern:
#   https://github.com/anthropics/claude-code/tree/main/.devcontainer
#
# Provides:
#   - Claude CLI (required by ClaudeSDKClient)
#   - Python 3.10 + all predicators dependencies
#   - claude-agent-sdk for in-process MCP tools
#   - Firewall restricting outbound to api.anthropic.com + GitHub
#
# Build from the repository root:
#   bash docker/build.sh
#
# Runtime bind-mounts (see docker_sandbox.py):
#   -v <repo>:/workspace:ro     read-only predicators source
#   -v <tmp_dir>:/data          shared data (pickled context + results)

FROM node:20

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

# ---------------------------------------------------------------------------
# System packages
#   - git, sudo, curl: basic dev tools
#   - jq, dnsutils (dig), aggregate, iptables, ipset, iproute2: firewall
#   - python3, python3-venv, python3-dev: Python runtime
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        sudo \
        curl \
        jq \
        iptables \
        ipset \
        iproute2 \
        dnsutils \
        aggregate \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# npm-global directory (mirrors official devcontainer setup)
# ---------------------------------------------------------------------------
RUN mkdir -p /usr/local/share/npm-global \
    && chown -R node:node /usr/local/share

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH="$PATH:/usr/local/share/npm-global/bin"

# ---------------------------------------------------------------------------
# Claude Code CLI (required by ClaudeSDKClient)
# ---------------------------------------------------------------------------
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# ---------------------------------------------------------------------------
# Firewall script — node user gets passwordless sudo for this one script only
# ---------------------------------------------------------------------------
COPY docker/init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh \
    && echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" \
       > /etc/sudoers.d/node-firewall \
    && chmod 0440 /etc/sudoers.d/node-firewall

# ---------------------------------------------------------------------------
# Claude config directory for the node user
# ---------------------------------------------------------------------------
RUN mkdir -p /home/node/.claude \
    && chown -R node:node /home/node/.claude

# ---------------------------------------------------------------------------
# Python environment — install predicators dependencies
#
# We copy setup.py and install deps. The actual source code is mounted
# read-only at /workspace at runtime, so code changes never require a
# rebuild — only PyPI dependency changes do.
# ---------------------------------------------------------------------------
WORKDIR /build
COPY setup.py ./
RUN mkdir -p predicators && touch predicators/__init__.py \
    && pip install --no-cache-dir --break-system-packages -e . || true
RUN pip install --no-cache-dir --break-system-packages \
    claude-agent-sdk dill nest_asyncio

# ---------------------------------------------------------------------------
# Cleanup and create directories
# ---------------------------------------------------------------------------
RUN rm -rf /build \
    && mkdir -p /sandbox /data \
    && chmod 777 /sandbox /data

# ---------------------------------------------------------------------------
# Entrypoint: init firewall -> exec command
# ---------------------------------------------------------------------------
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV DEVCONTAINER=true
ENV CLAUDE_CONFIG_DIR=/home/node/.claude
ENV NODE_OPTIONS="--max-old-space-size=4096"

USER node
WORKDIR /sandbox

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
