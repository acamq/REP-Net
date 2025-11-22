FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# ---- Build-time deps install ----
# Only copy requirements.txt so image doesn't need rebuilding for pure code changes
WORKDIR /opt/repnet
COPY requirements.txt /opt/repnet/requirements.txt

RUN python3 -m venv /opt/repnet-venv --system-site-packages && \
    . /opt/repnet-venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /opt/repnet/requirements.txt

# ---- Runtime entrypoint script ----
WORKDIR /workspace
COPY start-repnet.sh /start-repnet.sh
RUN chmod +x /start-repnet.sh

# Start RunPod stack (sshd etc.), then our script
CMD ["/bin/bash", "-lc", "/start.sh & sleep 5 && /start-repnet.sh; wait"]
