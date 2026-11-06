# Generated kernel acceptance checklist

- [ ] Exact source/runtime commit and artifact hashes recorded
- [ ] Complete operator contract summarized before implementation
- [ ] Current `asctile` API and supported syntax only
- [ ] Host code contains no numerical fallback or `data_ptr()` extraction
- [ ] Physical/logical shapes, neutral tails, and UB estimate documented
- [ ] `python3 -m py_compile` and static contract checks pass
- [ ] Every case dispatch and unique specialization passes exact-runtime lowering
- [ ] Ordinary and adversarial camodel routes are compared to the golden
- [ ] Reviewer used a fresh session/worker; actual model/session IDs recorded without claiming a different model unless used
- [ ] CANNBench job URL/result is recorded when remote execution completes
