# Scheduled corrected-target GeLU evaluation

Requested 2026-09-07. Status: **FP32 accuracy remedy locally qualified;
final combined-package qualification pending**;
no new hardware job or credit expenditure. This run takes priority in the
existing CANNBench GeLU-first queue; it does not create a second queue.

The prior direct-Erf FP32 failure in [EXACT_CHECK.md](EXACT_CHECK.md) remains
historical evidence. The user's requested ops-nn investigation produced a
locally passing high-level remedy: [EXACT_REMEDY.md](EXACT_REMEDY.md).
Use [exact_fast_kernel.py](exact_fast_kernel.py) for FP32 exact, SHA256
`a9779265f28440e138d19c2db40bc5465fe5d5d4b9c89e33195733005eae6f33`;
retain [exact_kernel.py](exact_kernel.py) with FP32 intermediates for FP16/BF16
exact. Resume preparation of the **same one** combined submission, not an
immediate upload. Revalidate the final wrapper, runtime/ABI, source hashes,
all 20 official routes and hardware-compatible artifact before spending credit.

## Candidate

- Pin source and both local/evaluator runtimes to
  `adadd7d66ed0ee16d33d79487bf584899a26ef1e` for comparison with this experiment.
- Use the corrected target tanh formula, with coefficients `0.044715` and
  `-sqrt(8/pi)`. FP32 uses the already tested corrected target.
- FP16/BF16 use [promoted_kernel.py](promoted_kernel.py): copy input,
  cast to public `asctile.float32`, compute all intermediates in FP32, then
  cast back to the input dtype on output. No lower-level replacements.
- Relevant changed tanh cases: FP16 4/10/14/17; BF16 6/12/19. FP32 tanh
  5/8 remain in the evaluation.
- User scope extension: implement exact GeLU in the **same target-style tiled
  template and same submission**. For FP16/BF16 retain this campaign's newly
  tested high-level exact template (not a legacy low-level fallback):

  ```python
  x = row.to(asctile.float32)
  y = (x * 0.5) * (asctile.erf(x * 0.7071067811865476) + 1.0)
  # copy_out(y.to(row.dtype), ...) with verified tail handling
  ```

  Keep FP32 intermediates for FP16/BF16. Scaling x first addresses a known
  finite-positive overflow ordering hazard; it does NOT establish negative-tail
  accuracy or prevent cancellation in `1 + erf(...)`. Verify emitted ordering.
  Never substitute tanh for `approximate="none"`.
- FP32 exact uses the public-Erf central expression for x >= -3, and a
  six-level normal-tail continued fraction for x < -3, as implemented in
  `exact_fast_kernel.py`. Selected local configuration: tile 1024, unroll 1,
  reuse 1, static default, VF on, UB 49152 B. The dense, random-tail,
  switch-boundary, special/finite-limit and repeated eight-core tail probes
  pass. Matched warmed FP32 Model median is 4966 ticks, not a NPU speedup.
- Low-precision exact starting geometry: tile 8192, unroll 2, reuse 1,
  static default, VF off, matching its passing numerical probe. Do not
  force the FP32 tail divisions onto FP16/BF16 without measured benefit.
- Exact cases: FP16 1/7; BF16 3/9/16; FP32 2/11/13/15/18/20.
  Together with the nine tanh cases these are all 20 official cases and all
  six dtype/mode routes. The direct `exact_kernel.py` FP32 route remains
  rejected; do not confuse it with the new selected FP32 remedy.

## Before spending a credit

1. Read the current skill and all official GeLU task files. Reconcile current
   jobs, archive hashes and credits; do not assume the historical balance.
2. Prepare a separate submission candidate with `gelu(x, approximate="none")`
   and correct host dispatch. Preserve the original targets and diagnostic
   evidence. Record source, runtime, artifact and task hashes.
3. Measure warmed CaModel performance for FP16/BF16 with FP32 intermediates;
   the existing promoted checks establish sampled accuracy, not performance.
   Start from tile 8192/unroll 2/reuse 1/static default/VF off (131072 B UB).
   Compare bounded tile/launch and supported JIT alternatives on matched
   inputs, pruning UB overflow before execution. Keep synchronization enabled.
   Qualify the new exact template independently for each dtype, including
   requested/effective JIT settings and its own UB budget; do not inherit tanh
   measurements or assume tile 8192 fits exact. Keep the authored implementation
   concise and high-level, with no special-function replacement polynomials.
4. Recheck official tolerances, negative tails, NaN/Inf, zero, finite limits,
   non-aligned shapes, multicore partitioning and repeated execution of the
   final candidate. Compile all 20 official specializations. Distinguish
   reduced local probes from full official-shape hardware validation.
   For exact use independent `torch.nn.functional.gelu(..., approximate="none")`
   references (double and same dtype) and the official precision checker.
   Explicitly reproduce the previous FP32 exact cancellation failures associated
   with cases 11/20 and the large finite-positive ordering problem. Diagnose
   surviving failures as evidence, not automatically as compiler-pass defects.
5. Verify the shared high-level source gate, evaluator wheel/ABI and final
   archive contents. Obtain a read-only independent OpenCode review using a
   healthy available model; record evidence and independently adjudicate it.

## Hardware run and stop conditions

Allow at most **one new private GeLU-only diagnostic submission** after the
changed routes pass local qualification and credits are available. Preserve
the full 20-case GeLU contract, with the new high-level exact and corrected
tanh routes in one package. Do not silently insert a historical or low-level
fallback, omit failed exact cases, or misrepresent tanh as exact. If any new
route fails local accuracy/compile/ABI gates, record the blocker and do not spend
a credit on a knowingly failing package without further user direction.

The purpose is now to measure **both** target-style exact and corrected tanh,
including FP32 intermediates for low-precision inputs. Local correctness is a
submission gate; achieving a speculative >=1x local-to-hardware speedup is not.
Do not claim that known exact-mode failures are fixed until measured.

Check pending jobs and tag/hash immediately before upload. Persist returned
submission/job IDs before polling. On ambiguous acceptance, reconcile server
state instead of retrying blindly. With no credit, defer using the existing
queue; do not purchase credits. No duplicate submission or automatic retuning
submission is authorized by this plan.

## Deliverable

Update [README.md](README.md) with the final source, per-case shape/dtype/mode,
requested/effective JIT options, UB, local timing, hardware accuracy and latency,
reference speedup, exact runtime revisions and clickable CANNBench job links.
Report the new tanh routes separately from the new exact routes and whole-job
score. Resume only the previously authorized downstream queue after recording
this run's outcome. Do not commit, push, or post upstream comments without a
separate request.
