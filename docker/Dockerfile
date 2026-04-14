# pyasc-sim: CANN 9.0 + Python 3.11 + pyasc for simulator verification.
#
# Build:
#   docker build -f docker/Dockerfile -t pyasc-sim:py3.11 docker/
#
# Usage:
#   docker run --rm -v "$(pwd)":/workspace -w /workspace \
#     pyasc-sim:py3.11 \
#     python3.11 tests/tools/run_and_verify.py golden/kernels/abs_f16.py --mode simulator
#
# Push to ghcr.io:
#   docker tag pyasc-sim:py3.11 ghcr.io/aloschilov/pyasc-sim:py3.11
#   docker push ghcr.io/aloschilov/pyasc-sim:py3.11

FROM ascendai/cann:9.0.0-beta.2-910b-ubuntu22.04-py3.11

ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib:${LD_LIBRARY_PATH}

RUN pip install --no-cache-dir \
    "pyasc>=1.1.0" \
    "numpy<2.0" \
    "pytest>=7.0" \
    pyyaml

# Non-fatal smoke test: pyasc native extensions may need full CANN env
# at runtime; the critical verification happens via docker run.
RUN python3 -c "import numpy; print(f'numpy: {numpy.__version__}')" && \
    python3 -c "import asc; print(f'asc: {asc.__version__}')" || \
    echo "WARN: pyasc import failed during build (expected if CANN runtime not fully initialized). Will work at container runtime."

WORKDIR /workspace
CMD ["/bin/bash"]
