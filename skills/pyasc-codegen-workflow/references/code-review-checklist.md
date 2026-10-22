# Generated kernel acceptance checklist

- [ ] Exact source/runtime commit and artifact hashes recorded
- [ ] Complete operator contract summarized before implementation
- [ ] Current `asctile` API and supported syntax only
- [ ] Host code contains no numerical fallback or `data_ptr()` extraction
- [ ] Physical/logical shapes, neutral tails, and UB estimate documented
- [ ] `python3 -m py_compile` and static contract checks pass
- [ ] Every case dispatch and unique specialization passes exact-v2 lowering
- [ ] Ordinary and adversarial camodel routes are compared to the golden
- [ ] Independent reviewer used a different model/session
- [ ] CANNBench job URL/result is recorded when remote execution completes
