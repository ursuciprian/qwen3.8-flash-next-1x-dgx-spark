# k101 arm image: a shipped spark-vllm-b12x image + the six vLLM files of ursuciprian/vllm exp/k101-gdn-views
# (VLLM_COMMIT), copied over the installed package. Fail-closed: every file the overlay replaces must still be the
# 5dad364d version (pin.sha256) and the result must be the branch version (patched.sha256). The 1x v2.3.0 and the 2x
# v2.1.0 images both serve vLLM 5dad364d for these files (the 2x pin d21d7ade changes none of them).
#   docker build --build-arg BASE=<shipped image> -t <tag> overlay/
ARG BASE
FROM ${BASE}
COPY vllm/ /tmp/k101/vllm/
COPY pin.sha256 patched.sha256 VLLM_COMMIT /tmp/k101/
RUN set -e; D=$(python3 -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('vllm').origin)))"); \
    cd "$D"; sha256sum -c /tmp/k101/pin.sha256; cp -r /tmp/k101/vllm/. "$D/vllm/"; sha256sum -c /tmp/k101/patched.sha256; \
    python3 -m compileall -q $(awk '{print $2}' /tmp/k101/patched.sha256); \
    printf '%s %s\n' "$(cat /tmp/k101/VLLM_COMMIT)" "$D" > /opt/k101-overlay; rm -rf /tmp/k101
