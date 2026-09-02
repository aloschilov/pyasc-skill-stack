Static checks failed on candidate.py:
- banned torch call: torch.zeros() — torch is allowed only for ['empty', 'empty_like']; all math must be asc2 kernels

Fix these and overwrite candidate.py.