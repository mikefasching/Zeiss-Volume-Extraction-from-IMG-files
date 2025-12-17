#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# SimpleITK is optional; if missing we still write .npy outputs
try:
    import SimpleITK as sitk  # type: ignore
    HAS_SITK = True
except Exception:
    HAS_SITK = False


def infer_shape_from_filesize(n_bytes: int):
    """
    Infer (d,h,w) from file size for known Zeiss export shapes.
    Zeiss .img is a raw uint8 dump => n_bytes == d*h*w.
    """
    known = [
        (200, 1024, 200),  # Macular Cube 200x200 (common)
    ]
    for d, h, w in known:
        if d * h * w == n_bytes:
            return (d, h, w)
    return None


def read_img(
    p: Path,
    size_pix_zyx=(200, 1024, 200),
    size_mm_zyx=(6, 2, 6),
):
    """
    Reads a raw Zeiss Cirrus OCT .img file (uint8 dump) -> (vol, spacing_zyx, sitk_img|None)
    vol shape: (z,y,x), dtype uint8
    """
    img_bytes = np.fromfile(p, dtype=np.uint8)

    d, h, w = size_pix_zyx
    expected = d * h * w
    if img_bytes.size != expected:
        raise ValueError(
            f"Size mismatch for {p}:\n"
            f"  got    {img_bytes.size} bytes\n"
            f"  expect {expected} bytes for shape {size_pix_zyx}\n"
            f"  (hint: this file might not be a 200x200 cube or has different dimensions)"
        )

    spacing_zyx = np.array(size_mm_zyx, dtype=np.float32) / np.array(size_pix_zyx, dtype=np.float32)

    vol = []
    hw = h * w
    for bscan_nr in range(d):
        s = hw * bscan_nr
        e = s + hw
        bscan = img_bytes[s:e].reshape((h, w))[::-1, ::-1]
        vol.append(bscan)

    vol = np.stack(vol, axis=0)  # (z,y,x)

    sim = None
    if HAS_SITK:
        sim = sitk.GetImageFromArray(vol)
        sim.SetSpacing(tuple(float(x) for x in spacing_zyx[::-1]))  # SITK spacing is (x,y,z)

    return vol, spacing_zyx, sim


def relative_under_images(img_path: Path):
    """
    Return the relative path under the first 'images' folder:
      .../images/<site>/<patient>/V3/.../DATAFILES/E223/file.img
    becomes:
      <site>/<patient>/V3/.../DATAFILES/E223/file.img

    If 'images' is not found, fallback to just the filename.
    """
    parts = img_path.parts
    try:
        idx = parts.index("images")
        rel_parts = parts[idx + 1 :]  # everything after 'images'
        return Path(*rel_parts)
    except Exception:
        return Path(img_path.name)


def safe_leaf_foldername(filename: str) -> str:
    """
    We store outputs under a folder named after the .img file.
    Make it filesystem-safe while keeping it readable.
    """
    return re.sub(r"[^\w\-.]+", "_", filename)


def process_one(
    img_path: Path,
    out_root: Path,
    default_mm=(6, 2, 6),
    force_shape=None,
    dry_run=False,
    overwrite=False,
):
    n_bytes = img_path.stat().st_size
    inferred = infer_shape_from_filesize(n_bytes)

    if force_shape is not None:
        shape = force_shape
    elif inferred is not None:
        shape = inferred
    else:
        shape = (200, 1024, 200)  # fallback

    # ✅ Recreate exact folder structure relative to /images
    rel = relative_under_images(img_path)  # includes filename at the end
    rel_parent = rel.parent               # everything except filename
    leaf_dir = safe_leaf_foldername(rel.name)

    out_dir = out_root / rel_parent / leaf_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.json"
    vol_npy = out_dir / "vol.npy"
    spacing_npy = out_dir / "spacing_zyx.npy"
    nii_path = out_dir / "vol.nii.gz"

    already_done = meta_path.exists() and vol_npy.exists() and spacing_npy.exists()
    if already_done and not overwrite:
        return "SKIP", str(img_path)

    if dry_run:
        return "DRYRUN", str(img_path)

    try:
        vol, spacing_zyx, sim = read_img(
            img_path,
            size_pix_zyx=shape,
            size_mm_zyx=default_mm,
        )

        np.save(vol_npy, vol)
        np.save(spacing_npy, spacing_zyx)

        sitk_written = False
        if HAS_SITK and sim is not None:
            sitk.WriteImage(sim, str(nii_path))
            sitk_written = True

        meta = {
            "input_path": str(img_path),
            "output_dir": str(out_dir),
            "relative_under_images": str(rel),
            "shape_zyx": list(map(int, shape)),
            "spacing_zyx_mm": [float(x) for x in spacing_zyx],
            "filesize_bytes": int(n_bytes),
            "sitk_written": sitk_written,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return "OK", str(img_path)

    except Exception as e:
        (out_dir / "error.txt").write_text(str(e) + "\n")
        return "ERR", f"{img_path} :: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Base images directory, e.g. /hddstore/.../images",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for extracted volumes (structure recreated under here)",
    )
    ap.add_argument(
        "--site",
        type=str,
        default="all",
        help="Site id (e.g. 001) or 'all'",
    )
    ap.add_argument(
        "--version",
        type=str,
        default="V3",
        help="Only process this visit/version folder (default: V3)",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.img",
        help="Glob pattern (default: *.img). You can pass '*200x200*.img'.",
    )
    ap.add_argument(
        "--max",
        type=int,
        default=0,
        help="Optional cap on number of files to process (0 = no cap)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be processed, but do nothing",
    )
    ap.add_argument(
        "--force-shape",
        type=str,
        default="",
        help="Force shape as 'd,h,w' (overrides inference). Example: 200,1024,200",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs instead of skipping",
    )
    args = ap.parse_args()

    base = args.base
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    force_shape = None
    if args.force_shape.strip():
        d, h, w = [int(x) for x in args.force_shape.split(",")]
        force_shape = (d, h, w)

    # Sites to iterate
    if args.site.lower() == "all":
        sites = sorted([p.name for p in base.iterdir() if p.is_dir()])
    else:
        sites = [args.site]

    grand_ok = grand_skip = grand_err = 0
    grand_total = 0

    for site_id in sites:
        site_dir = base / site_id
        if not site_dir.exists():
            print(f"[WARN] site dir not found: {site_dir}", file=sys.stderr, flush=True)
            continue

        patient_dirs = [p for p in site_dir.iterdir() if p.is_dir()]
        print(f"[SITE {site_id}] patients found: {len(patient_dirs)}", flush=True)

        for patient_dir in sorted(patient_dirs):
            patient_id = patient_dir.name
            vdir = patient_dir / args.version
            if not vdir.exists():
                continue

            # Only under V3/sdoct_cirrus
            cirrus_dir = vdir / "sdoct_cirrus"
            if not cirrus_dir.exists():
                continue

            # Only filenames containing "200x200"
            img_paths = []
            for p in cirrus_dir.rglob(args.pattern):
                if p.is_file() and ("200x200" in p.name):
                    img_paths.append(p)

            if not img_paths:
                continue

            print(f"[SITE {site_id} | PATIENT {patient_id}] files to process: {len(img_paths)}", flush=True)

            ok = skip = err = 0
            for i, img_path in enumerate(sorted(img_paths), 1):
                grand_total += 1

                print(
                    f"[SITE {site_id} | PATIENT {patient_id}] "
                    f"[{i}/{len(img_paths)}] processing: {img_path.name}",
                    flush=True,
                )

                status, msg = process_one(
                    img_path=img_path,
                    out_root=out,
                    default_mm=(6, 2, 6),
                    force_shape=force_shape,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )

                if status == "OK":
                    ok += 1
                    grand_ok += 1
                elif status == "SKIP":
                    skip += 1
                    grand_skip += 1
                elif status == "ERR":
                    err += 1
                    grand_err += 1

                print(f"[{status}] {msg}", flush=True)

                if args.max and grand_total >= args.max:
                    print(f"[INFO] Hit --max {args.max}, stopping early.", flush=True)
                    print(f"[DONE] OK={grand_ok} SKIP={grand_skip} ERR={grand_err} TOTAL={grand_total}", flush=True)
                    return

            print(
                f"[SITE {site_id} | PATIENT {patient_id} SUMMARY] OK={ok} SKIP={skip} ERR={err}",
                flush=True,
            )

    print(f"[DONE] OK={grand_ok} SKIP={grand_skip} ERR={grand_err} TOTAL={grand_total}", flush=True)


if __name__ == "__main__":
    main()
