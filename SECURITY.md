# Security Policy

## Supported Versions

The maintained line is the current `v1` source tree. Security and correctness fixes should target this line unless a release branch is created later.

## Reporting A Vulnerability

Please report suspected vulnerabilities, unsafe batch-execution behavior, or accidentally committed credentials to the project maintainers privately when possible. Include:

- affected file and command,
- input data shape or minimal reproduction steps,
- expected and actual behavior,
- environment details such as Python, PyTorch, CUDA, and pymatgen versions.

## Operational Notes

The refinement scripts are designed for local data processing and do not require network access at runtime. Batch execution uses `subprocess.run()` with a list of arguments and `shell=False` by default. Keep this property when modifying the batch runner.
