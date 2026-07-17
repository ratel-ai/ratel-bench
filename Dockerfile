# Slim toolchain image for the ratel-bench AWS CodeBuild pipeline (RAT-374).
#
# The image carries ONLY the stable toolchain + the bge-small embedding model.
# The repo code + buildspec come fresh from GitHub each build, and test-data +
# result caches come from S3 — so this image rebuilds rarely (only on a toolchain
# or model bump), while day-to-day pipeline changes ship via `git push`.
#
# Why the full Rust toolchain (not a prebuilt binary): every run does `version-set`
# (swaps the ratel-ai-core dependency) then `cargo run`, so cargo must be present
# to recompile the retrieval crate per version. Node/pnpm run the TS harness via tsx.
#
# Target arch: linux/arm64 (CodeBuild ARM_CONTAINER / Graviton). Build with:
#   docker buildx build --platform linux/arm64 -t <ecr>/ratel-bench:latest .

FROM node:24-bookworm

# ── System deps (build toolchain + git/ssh for version-set + python for the HF pull)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl git ca-certificates pkg-config \
      python3 python3-pip openssh-client unzip \
 && rm -rf /var/lib/apt/lists/*

# ── AWS CLI v2 (buildspec uses it for ssm/s3), arch-aware
RUN ARCH="$(uname -m)"; \
    case "$ARCH" in aarch64) A=aarch64;; x86_64) A=x86_64;; *) echo "unsupported $ARCH" && exit 1;; esac; \
    curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-${A}.zip" -o /tmp/awscliv2.zip \
 && unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscliv2.zip

# ── Rust toolchain (edition 2024 needs >= 1.85)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.90.0 --profile minimal

# ── pnpm via corepack (pinned to the repo's packageManager)
RUN corepack enable && corepack prepare pnpm@10.28.2 --activate

# ── Bake the bge-small embedding model into the HF cache (semantic/hybrid retriever)
ENV HF_HOME=/opt/hf-cache
RUN pip3 install --no-cache-dir --break-system-packages "huggingface_hub[cli]" \
 && huggingface-cli download BAAI/bge-small-en-v1.5 > /dev/null \
 && pip3 uninstall -y huggingface_hub > /dev/null 2>&1 || true

CMD ["bash"]
