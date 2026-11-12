"""Build every translated vector specialization with the flags recorded by PyASC."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile


def build(entry):
    key, commands = entry
    sdk = os.environ['ASCEND_HOME_PATH']
    source = Path('csrc/ops/swi_glu/op_kernel')/(key+'.cpp')
    dest = Path('cann_bench/device')/(key+'.o')
    with tempfile.TemporaryDirectory() as tmp:
        src, obj = Path(tmp)/'input.cce', Path(tmp)/'output.o'
        src.write_bytes(source.read_bytes())
        for command in commands:
            args = [arg.replace('@SDK@',sdk).replace('@SRC@',str(src)).replace('@OBJ@',str(obj)) for arg in command]
            subprocess.run(args, check=True)
        dest.write_bytes(obj.read_bytes())


if __name__ == '__main__':
    Path('cann_bench/device').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(build,json.loads(Path('flags.json').read_bytes()).items()))
