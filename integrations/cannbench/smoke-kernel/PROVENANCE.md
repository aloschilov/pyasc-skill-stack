# Reference candidate

This small permanent regression fixture was extracted unchanged from the
pre-existing `cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl` runtime/candidate bundle
(SHA256 `357f3d10961b2ad1878537b28563f6889931c1fd6b7a01fea32e150d9cd95333`).
That bundle records PyASC source commit
`030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d` in its metadata.

It is a historical candidate, not fresh model generation. The kernel uses
public `asctile` operations and performs numerical work on the NPU; the wrapper
handles allocation, layout and dispatch. Benchmark-specific correctness and
performance must be established by the live service, not inferred from this
provenance or from another runtime's simulator pass.
