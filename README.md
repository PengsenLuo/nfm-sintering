# NFM Sintering PINN — analysis code and data

Analysis pipeline, models, and data behind a study of how sintering process
parameters (temperature, ramp rate, holding time) and precursor particle
size interact to determine the morphology and packing (compacted) density of
NaNi₁/₃Fe₁/₃Mn₁/₃O₂ (NFM), a sodium-ion battery layered-oxide cathode
material. The dataset is a full-factorial design of 27 process conditions
(3 temperatures × 3 ramp rates × 3 holding times) × 3 precursor particle-size
architectures (S / M / L), 81 samples co-fired in the same furnace batches,
characterized by laser diffraction, SEM, XRD, and compacted-density
measurement.

## Citation

> Pengsen Luo, Yunlong Zhang, Zhan Shen, Chengan Liao, Zi-Feng Ma\*.
> **Hierarchical Design and Manufacturing of Sodium-Ion Layered Oxide
> Cathode Powders via Scale-Selective Control of Precursor Architecture and
> Thermal History.** 

## Repository structure

```
src/nfm/            Installable Python package: data processors (XRD/SEM/
                     PSD/compacted-density), feature engineering, models
                     (baselines + a physics-informed neural net), cross-
                     validation, statistics.
scripts/             Numbered pipeline scripts (data processing → feature
                     building → modeling → statistics → figures), plus
                     scripts/figs/ for figure generation.
tests/               pytest suite (130 tests).
configs/config.yaml  Hyperparameters, physical constants, instrument
                     calibration values, column-role declarations.
src/nfm/schema.py    Single source of truth for every data column: name,
                     unit, role (input vs. target), and processing layer.
data/raw/            Raw instrument output (see "Data" below — not all of
                     it is included in this repository).
data/interim/        Intermediate processing products (per-instrument
                     feature tables, diagnostics).
data/processed/      master_table.csv — the single entry point joining all
                     81 samples' inputs and measured targets. Everything in
                     this repository that reports a number derived from it.
source_data/         Journal-style Source Data: one CSV per main-text and
                     supplementary figure panel, plus MANIFEST.csv mapping
                     each file to its figure/panel.
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` gives loose lower bounds. `requirements-lock.txt` is the
exact environment (Python 3.11.15, Windows) the numbers in the paper were
produced and verified in — use it if you need byte-for-byte reproduction:

```bash
pip install -r requirements-lock.txt
```

`torch`/`torchvision` in the lock file are CUDA 12.1 wheels; on a CPU-only
machine install the CPU build instead (see comments in that file). GPU
support is only needed for the optional SAM-based SEM segmentation path
(see "Optional: SAM segmentation" below) — the core processing pipeline,
modeling, and test suite run on CPU.

### Running the tests

```bash
pytest tests/ -v
```

All 130 tests pass against `requirements-lock.txt`.

## Reproducing the analysis

The scripts are numbered in pipeline order. Each data processor only reads
`data/raw/` and writes `data/interim/`; `05_integrate.py` joins everything
(plus within-furnace difference quantities) into `data/processed/master_table.csv`,
which is the only input every downstream script needs.

```bash
python scripts/01_process_psd.py       # laser diffraction -> D10/D50/D90/Span + bimodal fit
python scripts/02_process_sem.py       # SEM -> secondary-particle size, shape descriptors
python scripts/03_process_xrd.py       # XRD -> D_XRD, Williamson-Hall, lattice parameters
python scripts/04_process_density.py   # compacted density
python scripts/05_integrate.py         # merge + within-furnace memory indices -> master_table.csv
python scripts/06_build_features.py    # physical derived features (thermal exposure Theta, etc.)
python scripts/07_train_eval.py --cv loco   # baselines + physics-constrained model, LOCO (27-fold)
```

Figure- and table-generation scripts live under `scripts/figs/` (exploratory)
and `scripts/figs/journal/` (the journal-submission figure set); each one
reads `data/processed/master_table.csv` and/or the relevant `data/interim/`
file and writes to `reports/figures/` (not included in this repository — the
rendered figures themselves are the paper's job, not this code's).

`source_data/` contains the exact numbers behind every published figure
panel; if you just want to check a reported number rather than rerun the
pipeline, start there — `source_data/MANIFEST.csv` maps each file to its
figure/panel and lists row/column counts and a SHA-256 checksum.

### Smoke test

A quick way to confirm your environment reproduces the paper's numbers:

```python
import pandas as pd
df = pd.read_csv("data/processed/master_table.csv")
print(df.groupby("precursor")["D_sec"].mean())
# S  ~4.61, M  ~10.33, L  ~10.46 (um)
```

## Data

| Directory | Contents | Included here? |
|---|---|---|
| `data/raw/xrd/` | Raw diffractograms (text) | Yes |
| `data/raw/psd/` | Laser-diffraction CSV exports | Yes |
| `data/raw/density/` | Compacted-density measurement log | Yes |
| `data/raw/sem/` | ~970 raw SEM TIFF images | **No** — see below |
| `data/interim/` | Per-instrument feature tables and processing diagnostics | Yes (image-only diagnostic subfolders, e.g. segmentation overlay PNGs, are excluded — they are not needed to reproduce any reported number) |
| `data/processed/master_table.csv` | The 81-sample joined table every reported number derives from | Yes |
| `source_data/` | Per-figure-panel Source Data (see above) | Yes |

**Raw SEM images are not included** (roughly 760 MB across ~970 TIFF files,
which would make this repository unreasonably large to clone). If you need
them to rerun `scripts/02_process_sem.py` from raw images, please contact
the corresponding author. All numbers derived from these images
(`data/interim/sem_features.csv`, `data/interim/sem_shape_descriptors.csv`,
and everything downstream) are included, so nothing that feeds
`master_table.csv` or `source_data/` requires the raw images.

### Optional: SAM segmentation

Primary-particle segmentation on the 5000× SEM images can optionally use
Meta's [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
instead of classical watershed segmentation (see `configs/config.yaml`,
`instruments.sem.sam`). The SAM checkpoint weights (up to 2.4 GB) are **not
included in this repository** — download them from the official release and
place them where `configs/config.yaml`'s `sam.checkpoint` path expects
them (default: `src/nfm/models/`):

- ViT-B (~375 MB, faster, slightly lower accuracy):
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- ViT-H (~2.4 GB, default, highest accuracy):
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

This is optional: `sam.enabled: false` in the config falls back to the
classical segmentation path, which is what most of the frozen
`data/interim/` feature tables in this repository were produced with for
the 500× secondary-particle line (see `src/nfm/data_processing/sem_processor.py`
docstrings for which measurement uses which path).

## Known limitations

- **The physics-informed model (PINN) does not beat a simple Ridge
  regression baseline on 6 of 7 target quantities.** This is reported as-is
  rather than tuned to pass a threshold — see the module docstring of
  `src/nfm/models/pinn.py` and `src/nfm/dxrd_baseline.py` for the numbers
  and the reasoning. The activation-energy parameter `Q` in the model is
  not identifiable from this dataset (see `training.pinn` comments in
  `configs/config.yaml`).
- Williamson–Hall grain-size/microstrain values (`D_XRD_WH`, `microstrain_WH`)
  have poor fit stability (R² 0.16–0.40 across samples) and are retained
  only as a qualitative cross-check, not a quantitative result — see
  `configs/config.yaml` and `src/nfm/data_processing/xrd_processor.py`.
- XRD/lattice-parameter analysis uses only the 51 samples run on one of the
  two instruments (see `src/nfm/nano_layer.py`); the other 30 samples (run
  on a second instrument) are retained in the data for audit purposes but
  excluded from that analysis for documented reasons (see that module).

## License

Code (`src/`, `scripts/`, `tests/`, `configs/`) is MIT-licensed — see
`LICENSE`. Data (`data/`, `source_data/`) is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see `data/LICENSE`.
