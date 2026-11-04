# Delivery roadmap

The supported path is OpenCode with PyASC skills, from a full checkout or linked
into a project. Use the pinned simulator for local checks and CANNBench for
external hardware measurements. The [recorded one-operator result](hardware-lane.md)
closes a bounded generation-to-measurement loop, not a performance campaign.

Next work, in order:

1. Automate fresh-session review and bounded tuning handoffs without losing the
   existing caller-managed skill loop; [issue #20](https://github.com/aloschilov/pyasc-skill-stack/issues/20)
   defines acceptance. Keep actual model/session and candidate provenance.
2. Add another operator recipe only with a task contract, adversarial numerical
   preflight and explicit qualification. Shared CI should not gain operator-specific
   branches. Unsupported recipes must fail before model or hardware use.
3. Improve measured performance only after correctness. Retain a correct incumbent,
   change one justified lever, and evaluate an explicitly authorized candidate
   within available CANNBench quota. A hint in a skill is not a measured speedup.

Keep software checks independent of hardware availability. Require current-head
PR gates; retain manifests, logs and candidates outside checkout, with durable
links/checksums for accepted qualification. No SSH backend, broad nightly sweep
or automatic credit-consuming retry is required for this path.
