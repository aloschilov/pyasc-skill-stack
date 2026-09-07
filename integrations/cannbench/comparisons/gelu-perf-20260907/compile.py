"""Compile every official dispatch and preserve translated source for analysis."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3] / 'integrations/cannbench/workers'))
import local_compile_gate as gate
from asc.runtime.compiler import Compiler

gate.PYASC_COMMIT = '0a631f70968c3cb7c33ce45330a85768dd5a6f06'
original = Compiler.run_translation
counter = 0
out = ROOT / 'translated' / Path(sys.argv[sys.argv.index('--candidate')+1]).parent.name
out.mkdir(parents=True, exist_ok=True)
def translated(self, module):
    global counter
    text = original(module)
    (out/f'{counter:02d}.cpp').write_text(text)
    counter += 1
    return text
Compiler.run_translation = translated
if __name__=='__main__': raise SystemExit(gate.main())
