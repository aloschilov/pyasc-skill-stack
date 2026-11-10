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

For SwiGLU, cast the two planes to FP32. If this pin has no public local slice,
build a row mask by concatenating [1, 256] zeros and ones, compare to 0.5,
select SiLU of the gate plane and the unmodified value plane with `where`,
then use `reduce_prod(selected, 0)`, cast to BF16 and reshape to [256] for
the store. Keep intermediates local. Never silently use host tensor indexing
or an unqualified gather on the last axis.

Assign whole tiles using the active vector grid and a block-stride loop.
The tile and transpose dimensions form one layout: deriving a new tile size
from UB alone does not preserve the permutation. Check compiled UB against the
active capacity. Validate pair order on varying inputs, a partial tile,
nonfinite lanes and the output write boundary before timing on hardware.
