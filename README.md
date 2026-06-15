# yfs_XRD_refinement

Version: v1

This project provides automated refinement tools for XRD patterns. The code is organized as an installable Python package under `src/yfs_xrd_refinement/`, while the original root-level scripts remain as compatibility wrappers.

Main workflows:

- `yfs_xrd_refinement.standard`: standard multi-stage automated refinement.
- `yfs_xrd_refinement.qlearning`: a Q-Learning enhanced version based on the standard workflow, intended for more adaptive parameter search.
- `yfs_xrd_refinement.batch`: batch runner for opXRD-style pattern folders.

Both scripts read an experimental `.xy` pattern, a main-phase CIF file, and an optional impurity-phase CIF folder. They perform candidate impurity screening, phase-combination search, peak-shape fitting, unit-cell refinement, atomic-position refinement, occupancy refinement, preferred-orientation refinement, and final result export.

## Project Status

- License: MIT. See `LICENSE`.
- Current release line: `v1`.
- Maintenance status: maintained as a research/engineering tool. Issues and pull requests for reproducibility, examples, tests, and bug fixes are welcome.
- Scope: this repository contains executable scripts, documentation, examples, and XRD/CIF data used for demonstration and validation.



## Repository Layout

```text
yfs_XRD_refinement/
  src/yfs_xrd_refinement/
    standard.py          # standard multi-stage refinement implementation
    qlearning.py         # Q-Learning enhanced implementation
    batch.py             # batch execution helper
    cli.py               # console entry points
  yfs_XRD.py             # compatibility wrapper for standard.py
  QL_yfs_XRD.py          # compatibility wrapper for qlearning.py
  parallel_batch_refine.py
  tests/
  pyproject.toml
```

```bash
# Optional editable install for console scripts
pip install -e .
```

## Installation

It is recommended to use an isolated Python environment. Install dependencies with:

```bash
pip install -r requirements.txt
```

For a reproducible environment close to the one used during repository cleanup, see `requirements-lock.txt`. The locked PyTorch line may be CUDA-build specific; choose a CPU/CUDA PyTorch build that matches your machine if needed.

Main dependencies:

- `numpy`
- `scipy`
- `torch`
- `matplotlib`
- `pymatgen`

The program automatically checks whether CUDA is available. If CUDA is available, GPU will be used; otherwise, CPU will be used.

## Input Files

A typical sample folder should look like this:

```text
sample_dir/
  sample.xy
  main.cif
  impure_phase/
    impurity_1.cif
    impurity_2.cif
```

File descriptions:

- `.xy` file: experimental XRD pattern. It should contain at least two columns: the first column is `2Theta`, and the second column is intensity.
- Main-phase CIF: specified by `--main`, for example `main.cif`.
- Impurity-phase folder: specified by `--imp`, for example `impure_phase`. All `.cif` files in this folder are treated as candidate impurity phases.
- If `--xy` is not specified, the program automatically selects the first `.xy` file in the current directory after sorting by filename.

## Basic Usage

Run the standard refinement from a sample folder. The old script entry still works:

```bash
python yfs_XRD.py --xy sample.xy --main main.cif --imp impure_phase
```

The package entry is equivalent:

```bash
PYTHONPATH=src python -m yfs_xrd_refinement standard --xy sample.xy --main main.cif --imp impure_phase
```

After `pip install -e .`, you can also use the console script:

```bash
yfs-xrd-refine --xy sample.xy --main main.cif --imp impure_phase
```

Run the Q-Learning version:

```bash
python QL_yfs_XRD.py --xy sample.xy --main main.cif --imp impure_phase
# or, after editable install:
yfs-xrd-ql-refine --xy sample.xy --main main.cif --imp impure_phase
```

For single-phase refinement, the impurity folder can be omitted:

```bash
python yfs_XRD.py --xy sample.xy --main main.cif
```

## Common Arguments

Both scripts support:

```bash
--xy              Experimental pattern file in .xy format
--main            Main-phase CIF file
--imp             Folder containing impurity-phase CIF files
--num-workers     Number of parallel workers; defaults to CPU core count
--main-bias       Main-phase bias used only during phase-combination screening; default is 1.0
--stoich-phase    Reference phase for stoichiometry constraints, usually the main phase
--stoich          Target stoichiometry, for example "Li:6,S:5,P:1,Cl:1"
--lambda-stoich   Stoichiometry constraint strength; default is 0.5
```

`QL_yfs_XRD.py` additionally supports:

```bash
--wl              X-ray wavelength in Angstrom; default is 1.5406
```

Example:

```bash
python QL_yfs_XRD.py \
  --xy sample.xy \
  --main main.cif \
  --imp impure_phase \
  --wl 1.5406 \
  --num-workers 8 \
  --stoich "Li:6,S:5,P:1,Cl:1" \
  --lambda-stoich 0.5
```

## Output Files

Common output files include:

```text
yfsf_Refined.png          Final refinement plot
yfsf_Refined.xy           Experimental and fitted pattern data
yfsf_Refined.txt          Final report with Rwp, phase fractions, scale factors, TCH parameters, etc.
yfsf_refined_cifs/        Refined CIF files
refine_log.csv            Refinement process log
Rwp_curve.png             Rwp curve
phase_fraction_curve.png  Phase-fraction curve
scale_curve.png           Scale-factor curve
```

`yfs_XRD.py` also saves intermediate stage results, such as:

```text
stage1_output/
stage2_output/
```

These folders contain stage-wise plots, fitted curves, text reports, and CIF files.

## Which Script Should I Use?

Start with the standard version:

```bash
python yfs_XRD.py --xy sample.xy --main main.cif --imp impure_phase
```

If the standard version appears trapped in a local optimum, if parameter updates are unstable, or if you want to try reinforcement-learning based action selection, run:

```bash
python QL_yfs_XRD.py --xy sample.xy --main main.cif --imp impure_phase
```

`QL_yfs_XRD.py` usually performs a more complex search and may take longer to run.

## If Refinement Quality Is Poor

If the final Rwp is high, the background fit looks abnormal, the low-angle background is too strong, or the fitted curve is clearly distorted by background intensity, manually subtract the background first and then rerun the refinement using the background-subtracted `.xy` file.

Recommended workflow:

1. Use common XRD software or your own script to subtract the background from the raw pattern.
2. Export a new two-column `.xy` file. Keep the first column as `2Theta` and the second column as the background-subtracted intensity.
3. Rerun refinement with the new `.xy` file:

```bash
python yfs_XRD.py --xy sample_bg_removed.xy --main main.cif --imp impure_phase
```

If the result is still unsatisfactory after background subtraction, check whether the main-phase CIF is correct, whether the candidate impurity phases are complete, whether the wavelength is correct, and whether there is strong preferred orientation or systematic peak shift.

## Testing

Run the lightweight static smoke tests before publishing changes:

```bash
python -m unittest discover -s tests
python -m py_compile yfs_XRD.py QL_yfs_XRD.py parallel_batch_refine.py src/yfs_xrd_refinement/*.py
```

The current tests intentionally avoid requiring large XRD datasets or GPU hardware. Add numerical regression tests when stable reference outputs are available.

## Security And Batch Execution

- The scripts do not make external network requests during refinement.
- `parallel_batch_refine.py` launches refinement with an argument list and the default `shell=False`; do not replace this with shell command strings.
- Treat CIF, JSON, and XY files from untrusted sources as data inputs. Run large or unknown datasets in an isolated environment and review output logs.
- Report suspected vulnerabilities or accidentally committed secrets privately if possible; see `SECURITY.md`.

## Open Source Compliance

- Project license: MIT, declared in `LICENSE` and via SPDX headers in Python source files.
- Third-party dependency license summary: see `THIRD_PARTY_NOTICES.md`.
- Reproducibility reference: `requirements-lock.txt`.
- Audit response and remaining work: `OPEN_SOURCE_REVIEW_RESPONSE.md`.

## License

This project is released under the MIT License. See `LICENSE` for the full text.

## Notes

- `.xy` intensities are normalized internally.
- More candidate impurity phases increase the phase-combination search time and total refinement time.
- Avoid setting `--num-workers` higher than the machine can handle, as too many workers may cause high memory usage.
- The `--stoich` format must use English colons and commas, for example `"Li:6,S:5,P:1,Cl:1"`.
- If a CIF file does not contain `Uiso`, the program uses a default value and attempts to complete the field during CIF export.
