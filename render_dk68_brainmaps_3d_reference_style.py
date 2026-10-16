# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BASE = Path(r"C:\Users\laoma\.codex\visualizations\2026\09\14\01a09fbb-0280-7fb2-b300-2e243f47d187\timescale_power_fooof")
DK_DIR = BASE / "dk_brain_maps"
AGE_QC_DIR = BASE / "age_association_qc"
OUT = BASE / "dk_brain_maps_3d_reference_style"
OUT.mkdir(parents=True, exist_ok=True)
CAT12 = Path(r"C:\Users\laoma\.brainstorm\plugins\cat12\cat12")
SURF_DIR = CAT12 / "templates_surfaces_32k"
ANNOT_DIR = CAT12 / "atlases_surfaces_32k"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

NATURE_SEQ = LinearSegmentedColormap.from_list(
    "nature_timescale_seq",
    ["#484878", "#7884B4", "#E4CCD8", "#F0C0CC", "#F0E0D0"],
    N=256,
)
NATURE_DIV = LinearSegmentedColormap.from_list(
    "nature_age_beta_div",
    ["#3775BA", "#B4C0E4", "#F7F7F7", "#E9A6A1", "#B64342"],
    N=256,
)


def dk_labels():
    _, _, raw_names = nib.freesurfer.read_annot(str(ANNOT_DIR / "lh.aparc_DK40.freesurfer.annot"))
    names = [n.decode("utf-8").strip() for n in raw_names]
    keep = [i for i, n in enumerate(names) if n not in ("unknown", "corpuscallosum")]
    return keep


def load_hemi(hemi):
    surf = nib.load(str(SURF_DIR / f"{hemi}.inflated.freesurfer.gii"))
    verts = surf.darrays[0].data.astype(float)
    faces = surf.darrays[1].data.astype(np.int64)
    labels, _, _ = nib.freesurfer.read_annot(str(ANNOT_DIR / f"{hemi}.aparc_DK40.freesurfer.annot"))
    verts = verts.copy()
    verts -= verts.mean(axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(verts, axis=1), 99)
    verts /= scale
    return verts, faces, labels


def face_values(labels, faces, label_value):
    vertex_values = np.full(labels.shape, np.nan, dtype=float)
    for lab, val in label_value.items():
        vertex_values[labels == lab] = val
    vals = vertex_values[faces]
    return np.nanmean(vals, axis=1)


def shaded_facecolors(verts, faces, values, cmap, norm, missing=(0.70, 0.70, 0.70, 1.0)):
    base = cmap(norm(values))
    base[~np.isfinite(values)] = missing
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm = np.linalg.norm(normals, axis=1)
    ok = nrm > 0
    normals[ok] /= nrm[ok, None]
    light = np.array([-0.45, -0.35, 0.82])
    light /= np.linalg.norm(light)
    shade = np.clip(normals @ light, 0, 1)
    shade = 0.52 + 0.48 * shade
    colors = base.copy()
    colors[:, :3] *= shade[:, None]
    colors[:, :3] = np.clip(colors[:, :3] + 0.09, 0, 1)
    return colors


def style_3d_axis(ax, hemi, view):
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.16, 1.16, 0.72))
    ax.grid(False)
    ax.set_facecolor((1, 1, 1, 0))
    if hemi == "lh" and view == "lateral":
        ax.view_init(elev=0, azim=180, roll=0)
    elif hemi == "rh" and view == "lateral":
        ax.view_init(elev=0, azim=0, roll=0)
    elif hemi == "lh" and view == "medial":
        ax.view_init(elev=0, azim=0, roll=0)
    elif hemi == "rh" and view == "medial":
        ax.view_init(elev=0, azim=180, roll=0)
    lim = 0.92
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.78, 0.78)


def draw_hemi(ax, verts, faces, labels, label_value, cmap, norm):
    vals = face_values(labels, faces, label_value)
    fc = shaded_facecolors(verts, faces, vals, cmap, norm)
    mesh = Poly3DCollection(verts[faces], facecolors=fc, edgecolors="none", linewidths=0, antialiased=True)
    mesh.set_rasterized(True)
    ax.add_collection3d(mesh)


def render_plate(values68, out_stem, cmap_name="magma", diverging=False, colorbar=True):
    keep = dk_labels()
    values68 = np.asarray(values68, dtype=float)
    if diverging:
        lim = np.nanpercentile(np.abs(values68), 95)
        if not np.isfinite(lim) or lim <= 0:
            lim = np.nanmax(np.abs(values68))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    else:
        vmin, vmax = np.nanpercentile(values68, [5, 95])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = np.nanmin(values68), np.nanmax(values68)
        norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = NATURE_DIV if diverging else NATURE_SEQ

    lh_verts, lh_faces, lh_labels = load_hemi("lh")
    rh_verts, rh_faces, rh_labels = load_hemi("rh")
    lh_value = {lab: values68[i] for i, lab in enumerate(keep)}
    rh_value = {lab: values68[34 + i] for i, lab in enumerate(keep)}

    height = 5.75 if colorbar else 5.05
    fig = plt.figure(figsize=(7.2, height), facecolor="white")
    gs = fig.add_gridspec(3 if colorbar else 2, 2, height_ratios=([1, 1, 0.08] if colorbar else [1, 1]), hspace=-0.30, wspace=-0.22)
    axes = [fig.add_subplot(gs[0, 0], projection="3d"), fig.add_subplot(gs[0, 1], projection="3d"), fig.add_subplot(gs[1, 0], projection="3d"), fig.add_subplot(gs[1, 1], projection="3d")]
    panels = [("lh", "lateral", lh_verts, lh_faces, lh_labels, lh_value), ("rh", "lateral", rh_verts, rh_faces, rh_labels, rh_value), ("lh", "medial", lh_verts, lh_faces, lh_labels, lh_value), ("rh", "medial", rh_verts, rh_faces, rh_labels, rh_value)]
    for ax, (hemi, view, verts, faces, labels, label_value) in zip(axes, panels):
        draw_hemi(ax, verts, faces, labels, label_value, cmap, norm)
        style_3d_axis(ax, hemi, view)
    if colorbar:
        cax = fig.add_subplot(gs[2, :])
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, width=0.6, length=2.5)
    fig.subplots_adjust(left=-0.035, right=1.035, top=1.02, bottom=0.0)
    fig.savefig(OUT / f"{out_stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUT / f"{out_stem}.pdf", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def read_region_vector(csv_path, col):
    d = pd.read_csv(csv_path).sort_values("region")
    return d[col].to_numpy(dtype=float)


def main():
    jobs = []
    for site in ["All", "CAM_CAN", "OMEGA", "PANS"]:
        csv_path = DK_DIR / f"ComBat_timescale_DK68_{site}_68x1.csv"
        if csv_path.exists():
            jobs.append((f"ComBat_timescale_DK68_{site}_3d_ref_clean", read_region_vector(csv_path, "timescale_ms_mean"), "magma", False))
    age_qc = AGE_QC_DIR / "ComBat_timescale_age_association_DK68_QC_nonboundary_fitR2gt0.csv"
    if age_qc.exists():
        jobs.append(("ComBat_timescale_age_beta_DK68_QC_3d_ref_clean", read_region_vector(age_qc, "age_beta_ms_per_year_adjusted"), "coolwarm", True))
    for stem, values, cmap, diverging in jobs:
        render_plate(values, stem, cmap_name=cmap, diverging=diverging, colorbar=False)
        render_plate(values, stem + "_with_colorbar", cmap_name=cmap, diverging=diverging, colorbar=True)
        print("wrote", OUT / f"{stem}.png")


if __name__ == "__main__":
    main()



