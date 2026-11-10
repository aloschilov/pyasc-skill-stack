# Read interleaved BF16 pairs as one flat tile

When each output consumes two adjacent elements in contiguous storage, flatten
the outer dimensions into a pair count. Read a contiguous span starting at
`2 * first_pair`; write one contiguous result starting at `first_pair`. The
input and output valid extents are respectively twice the valid pair count and
the valid pair count. Align full tile boundaries and give the last partial
tile a single writer. Do not round up each individual operand within each row.

On pins without a qualified local unzip operation, this AscTile layout splits
256 BF16 pairs using two aligned transposes. Both transpose dimensions are
multiples of 16 BF16 elements. Reshaping the flat input directly to [2, 256]
would instead put the first half of the input in the first plane.

```python
flat = asctile.copy_in(x, [2 * first], [512], real_shape=[2 * valid], pad_value=0.0)
t = flat.reshape(16, 32).transpose(1, 0)
planes = t.reshape(16, 32).transpose(1, 0).reshape(2, 256)
```

See the [complete pair kernel](../templates/packed-bf16-swiglu.py) for the exact
public API calls. It selects SiLU of the gate plane and the original value
plane with a two-row mask, then reduces their product along the plane axis.
Its block-stride wrapper takes a first group, a count of groups and a single
compile-time valid extent. Launch the full groups with valid = 256 and, if
needed, the final group with its actual valid extent. Keep intermediates local.
Do not use host tensor indexing or an unqualified gather on the last axis.

Assign whole tiles using the active vector grid and a block-stride loop.
The tile and transpose dimensions form one layout: deriving a new tile size
from UB alone does not preserve the permutation. Check compiled UB against the
active capacity. The linked FP32-compute specialization used 23,104 UB bytes
for both 256 and 67 valid pairs on the 950PR at PyASC 6b3447b4. Recheck on
a different runtime or after changing the computation. Validate pair order on varying inputs, a partial tile,
nonfinite lanes and the output write boundary before timing on hardware.
