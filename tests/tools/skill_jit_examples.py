"""Extract complete JIT listings verbatim; inventory every exclusion explicitly."""
import hashlib
import json
from pathlib import Path
from check_skill_code_blocks import iter_blocks
from run_artifacts import ROOT

# These listings explain individual syntax constructs, not standalone kernels.
# New unclassified JIT listings fail closed rather than silently losing coverage.
EXCLUDED = {
    'skills/pyasc-syntax-constraints/SKILL.md': 'ellipsis body; syntax fragment',
    'skills/pyasc-syntax-constraints/references/unsupported-syntax.md': 'deliberately invalid counter-example',
    'skills/pyasc-syntax-constraints/references/supported-syntax.md': 'helper/caller fragment without imports, output or standalone launch contract',
    'skills/pyasc-syntax-constraints/references/constexpr-guide.md': 'conditional input fragment without imports or output',
}

# Parameters belong to a source listing, never its enumeration position.
SCALAR_ARGS = {
    'skills/pyasc-target-operator/SKILL.md': {'size': 1025, 'tile_size': 1024},
}


def extract_examples(directory):
    directory = Path(directory); directory.mkdir(parents=True)
    inventory = []; examples = []; seen_exclusions = set()
    exclusion_counts = {}
    for path in sorted((ROOT/'skills').rglob('*.md')):
        rel = str(path.relative_to(ROOT))
        for line, code, marker in iter_blocks(path.read_text()):
            if '@asctile.jit' not in code and '@asc.jit' not in code: continue
            reason = ('explicit '+marker) if marker else EXCLUDED.get(rel)
            if rel in EXCLUDED:
                seen_exclusions.add(rel)
                exclusion_counts[rel] = exclusion_counts.get(rel, 0) + 1
                if exclusion_counts[rel] != 1:
                    raise ValueError(f"Additional JIT block needs its own classification: {rel}:{line}")
            entry = {'file': rel, 'line': line, 'sha256': hashlib.sha256(code.encode()).hexdigest(),
                     'excluded': reason}
            inventory.append(entry)
            if reason: continue
            label = f'skill-jit-{len(examples)+1}'
            entry['label'] = label
            entry['scalar_args'] = SCALAR_ARGS.get(rel, {})
            output = directory/(label+'.py'); output.write_text(code+'\n')
            examples.append((label, output))
    if not examples or seen_exclusions != set(EXCLUDED):
        raise ValueError('Empty JIT coverage or stale exclusion inventory')
    (directory/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    return examples
