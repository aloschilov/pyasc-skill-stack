"""Create separate finite FP16 controls from the public case-7 schema."""
import csv
import json
from pathlib import Path

src = Path('generation/task')
for name, rows, half in [('near4',577,1536), ('near8',1154,768)]:
    dest = Path(name)
    task = dest/'generation/task'
    task.mkdir(parents=True)
    with (src/'cases.csv').open() as f:
        row, = (row for row in csv.DictReader(f) if row['case_id']=='7')
    row['input_shape'] = json.dumps([[rows, 2*half]])
    row['note'] = 'Synthetic adjacent shape for two-wave FP16 schedule comparison'
    with (task/'cases.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    (task/'golden.py').write_bytes((src/'golden.py').read_bytes())
    for pkg in ('base','adjacent','strided'):
        (dest/pkg).symlink_to(Path(pkg).resolve(),target_is_directory=True)
