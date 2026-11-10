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

For SwiGLU on pins without a public local slice, select the transformed gate
plane and the original value plane. On the tested AscTile API, concat joins
the two mask rows along axis zero and the reduction returns [1, 256]:

```python
xy = asctile.cast(planes, asctile.float32)
gate_row = asctile.full([1, 256], 0.0, asctile.float32)
value_row = asctile.full([1, 256], 1.0, asctile.float32)
is_gate = asctile.concat(gate_row, value_row) < 0.5
inv = asctile.div(1.0, asctile.add(asctile.exp(asctile.mul(xy, -1.0)), 1.0))
selected = asctile.where(is_gate, asctile.mul(xy, inv), xy)
product = asctile.cast(asctile.reduce_prod(selected, 0), asctile.bfloat16)
asctile.copy_out(product.reshape(256), y, [first], real_shape=[valid])
```

Keep intermediates local. Never silently use host tensor indexing or an
unqualified gather on the last axis.

Assign whole tiles using the active vector grid and a block-stride loop.
The tile and transpose dimensions form one layout: deriving a new tile size
from UB alone does not preserve the permutation. Check compiled UB against the
active capacity. Validate pair order on varying inputs, a partial tile,
nonfinite lanes and the output write boundary before timing on hardware.
