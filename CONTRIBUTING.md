# Contributing

Thank you for improving `yfs_XRD_refinement`. The project is a research-oriented XRD refinement tool, so changes should preserve numerical behavior unless the change is explicitly intended to alter the algorithm.

## Before Submitting

Run:

```bash
python -m unittest discover -s tests
python -m py_compile yfs_XRD.py QL_yfs_XRD.py parallel_batch_refine.py
```

For algorithm changes, also run at least one example folder and include the final Rwp and generated-output differences in the pull request notes.

## Change Guidelines

- Keep command-line interfaces backward compatible when possible.
- Avoid `shell=True` in batch execution.
- Document new data files, external snippets, or generated artifacts with provenance and license information.
- Prefer small refactors that reduce duplication between `yfs_XRD.py` and `QL_yfs_XRD.py` without changing refinement results.
