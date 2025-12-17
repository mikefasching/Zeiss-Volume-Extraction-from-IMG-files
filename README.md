# Zeiss Cirrus OCT `.img` Volume Extractor

This repository provides a lightweight Python tool to convert **Zeiss Cirrus OCT raw `.img` files** into structured, machine-learning–ready volumes.

The script is specifically designed for **Macular Cube 200×200** scans exported by Zeiss Cirrus devices and converts raw binary dumps into NumPy arrays, with optional medical-imaging output.

---

## Features

- Converts Zeiss Cirrus `.img` files (raw uint8 dumps) into 3D volumes
- Supports **Macular Cube 200×200** scans (default shape: `200 × 1024 × 200`)
- Outputs NumPy arrays for fast ML workflows
- Optionally writes NIfTI (`.nii.gz`) for medical imaging tools
- Preserves the **exact folder structure** relative to the `images/` directory
- Safe to re-run (already processed files are skipped by default)
- No proprietary dependencies

---

## Supported input data

### Scan type
- **Zeiss Cirrus OCT**
- **Macular Cube 200×200**

### Expected data layout

The script expects a directory structure that contains an `images/` folder, for example:

```
images/
└── <site_id>/
    └── <patient_id>/
        └── V3/
            └── sdoct_cirrus/
                └── .../
                    └── DATAFILES/
                        └── E###/
                            └── <scan_file>.img
```

Only files that satisfy **all** of the following conditions are processed:

- Located under `V3/sdoct_cirrus`
- Filename contains `200x200`
- File size matches the expected volume shape (or a forced shape is provided)

---

## Output structure

The output directory recreates the **exact folder hierarchy under `images/`**.

For an input file:

```
images/003/0015/V3/sdoct_cirrus/.../DATAFILES/E799/example.img
```

the outputs are written to:

```
<output_root>/
└── 003/
    └── 0015/
        └── V3/
            └── sdoct_cirrus/
                └── .../
                    └── DATAFILES/
                        └── E799/
                            └── example.img/
                                ├── vol.npy
                                ├── spacing_zyx.npy
                                ├── vol.nii.gz   (optional)
                                └── meta.json
```

The leaf directory is named after the original `.img` file (sanitized for filesystem safety).

---

## Output files

### `vol.npy`
- 3D NumPy array of shape `(z, y, x)`
- Default shape: `(200, 1024, 200)`
- Data type: `uint8`
- Intended for machine learning and numerical analysis

### `spacing_zyx.npy`
- NumPy array of length 3
- Voxel spacing in millimeters
- Order: `(z, y, x)`

### `vol.nii.gz` *(optional)*
- NIfTI medical imaging format
- Includes voxel spacing in the header

### `meta.json`
- Metadata and traceability information
- Original input path
- Output directory
- Volume shape and spacing
- Whether NIfTI output was written

---

## Requirements

- Python 3.7 or newer
- `numpy`
- Optional: `SimpleITK` (required only for `.nii.gz` output)

---

## Installation

Install the required dependencies using pip:

```
pip install numpy
pip install SimpleITK  # optional
```

---

## Usage

Run the script from the command line:

```
python extract_zeiss_volumes.py \
  --base /path/to/images \
  --out  /path/to/output_root
```

Optional arguments:

- `--site <id>`: process a single site (default: all)
- `--version <str>`: visit/version folder (default: `V3`)
- `--pattern <glob>`: filename pattern (default: `*.img`)
- `--dry-run`: list files without processing
- `--force-shape d,h,w`: override automatic shape inference
- `--overwrite`: reprocess files even if outputs exist

---

## Notes and limitations

- The script assumes Zeiss `.img` files are **raw uint8 binary dumps**
- Currently tuned for **Macular Cube 200×200** scans
- Other scan types require adding shapes to the code or using `--force-shape`
- This repository contains **code only** and does not include any patient data

---

## License

Choose an appropriate license (e.g. MIT or Apache-2.0) depending on your intended usage.
