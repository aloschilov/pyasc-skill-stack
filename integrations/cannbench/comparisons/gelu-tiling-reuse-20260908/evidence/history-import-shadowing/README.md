# Rejected harness attempts

These three attempts did not import the runtime or execute a kernel: an added
configuration-selector script was named `select.py`, shadowing Python's standard
library module during `subprocess` import. Renamed to `choose_config.py` and reran
the identical configurations in fresh processes. This is a harness defect, not a
pyasc/compiler failure. The failed records are retained and never used as timings.
