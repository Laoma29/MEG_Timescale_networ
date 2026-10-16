# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import nibabel as nib

BASE = Path(r'C:\Users\laoma\.codex\visualizations\2026\09\14\01a09fbb-0280-7fb2-b300-2e243f47d187\timescale_power_fooof')
IN_CSV = BASE / 'timescale_ComBat_model_ready_long.csv'
DK_SUMMARY = BASE / 'dk_brain_maps' / 'ComBat_timescale_DK68_All_68x1.csv'
OUT = BASE / 'age_association_qc'
OUT.mkdir(parents=True, exist_ok=True)
CAT12 = Path(r'C:\Users\laoma\.brainstorm\plugins\cat12\cat12')
SURF_DIR = CAT12 / 'templates_surfaces_32k'
ANNOT_DIR = CAT12 / 'atlases_surfaces_32k'

QC_QUERY = '(knee_at_lower_grid_bound == False) and (knee_at_upper_grid_bound == False) and (fit_r2 > 0)'


def fdr(p):
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan, dtype=float)
    if ok.sum():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q


def dk_region_names():
    _, _, names = nib.freesurfer.read_annot(str(ANNOT_DIR / 'lh.aparc_DK40.freesurfer.annot'))
    names = [n.decode('utf-8').strip() for n in names]
    keep_ids = [i for i, n in enumerate(names) if n not in ('unknown', 'corpuscallosum')]
    return keep_ids, [f'lh_{names[i]}' for i in keep_ids] + [f'rh_{names[i]}' for i in keep_ids]


def load_hemi(hemi):
    surf = nib.load(str(SURF_DIR / f'{hemi}.inflated.freesurfer.gii'))
    verts = surf.darrays[0].data.astype(float)
    faces = surf.darrays[1].data.astype(np.int64)
    labels, _, _ = nib.freesurfer.read_annot(str(ANNOT_DIR / f'{hemi}.aparc_DK40.freesurfer.annot'))
    return verts, faces, labels


def project_view(verts, hemi, view):
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    if view == 'lateral':
        return np.c_[y, z] if hemi == 'lh' else np.c_[-y, z]
    if view == 'medial':
        return np.c_[-y, z] if hemi == 'lh' else np.c_[y, z]
    raise ValueError(view)


def face_values(labels, faces, label_value):
    vertex_values = np.full(labels.shape, np.nan, dtype=float)
    for lab, val in label_value.items():
        vertex_values[labels == lab] = val
    fv = vertex_values[faces]
    return np.nanmean(fv, axis=1)


def plot_panel(ax, verts, faces, labels, hemi, view, label_value, cmap, norm):
    coords = project_view(verts, hemi, view)
    vals = face_values(labels, faces, label_value)
    colors = cmap(norm(vals))
    colors[np.isnan(vals)] = [0.86, 0.86, 0.86, 1.0]
    pc = PolyCollection(coords[faces], facecolors=colors, edgecolors='none', linewidths=0, antialiased=False)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.axis('off')


def render_clean_brain(values68, out_png, diverging=False):
    keep_ids, _ = dk_region_names()
    vals68 = np.asarray(values68, dtype=float)
    if diverging:
        lim = np.nanpercentile(np.abs(vals68), 95)
        if not np.isfinite(lim) or lim <= 0:
            lim = np.nanmax(np.abs(vals68))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        cmap = plt.get_cmap('coolwarm')
    else:
        vmin, vmax = np.nanpercentile(vals68, [5, 95])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = np.nanmin(vals68), np.nanmax(vals68)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('magma')
    lh_verts, lh_faces, lh_labels = load_hemi('lh')
    rh_verts, rh_faces, rh_labels = load_hemi('rh')
    lh_value = {lab: vals68[i] for i, lab in enumerate(keep_ids)}
    rh_value = {lab: vals68[34 + i] for i, lab in enumerate(keep_ids)}
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2))
    for ax in axes.ravel():
        ax.set_facecolor('white')
    plot_panel(axes[0,0], lh_verts, lh_faces, lh_labels, 'lh', 'lateral', lh_value, cmap, norm)
    plot_panel(axes[1,0], lh_verts, lh_faces, lh_labels, 'lh', 'medial', lh_value, cmap, norm)
    plot_panel(axes[0,1], rh_verts, rh_faces, rh_labels, 'rh', 'lateral', rh_value, cmap, norm)
    plot_panel(axes[1,1], rh_verts, rh_faces, rh_labels, 'rh', 'medial', rh_value, cmap, norm)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01, hspace=0.01)
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)


def region_age_models(df):
    _, names = dk_region_names()
    rows = []
    for region in range(1, 69):
        sub = df[(df.region == region) & np.isfinite(df.age) & np.isfinite(df.timescale_ms)].copy()
        if len(sub) < 30 or sub['age'].nunique() < 5:
            rows.append({'region': region, 'region_name': names[region-1], 'n_qc': len(sub)})
            continue
        pear_r, pear_p = stats.pearsonr(sub['age'], sub['timescale_ms'])
        spear_r, spear_p = stats.spearmanr(sub['age'], sub['timescale_ms'])
        try:
            m = smf.ols('timescale_ms ~ age + C(sex) + C(site)', data=sub).fit(cov_type='HC3')
            beta = m.params.get('age', np.nan); se = m.bse.get('age', np.nan)
            t = m.tvalues.get('age', np.nan); p = m.pvalues.get('age', np.nan)
            ci = m.conf_int().loc['age'].to_numpy() if 'age' in m.params.index else [np.nan, np.nan]
            r2 = m.rsquared
        except Exception:
            beta=se=t=p=r2=np.nan; ci=[np.nan,np.nan]
        rows.append({
            'region': region, 'region_name': names[region-1], 'n_qc': len(sub),
            'subject_n_qc': sub.subject_id.nunique(),
            'age_beta_ms_per_year_adjusted': beta,
            'age_beta_ci_low': ci[0], 'age_beta_ci_high': ci[1],
            'age_se': se, 'age_t': t, 'age_p_adjusted_model': p,
            'model_r2': r2,
            'pearson_r_raw': pear_r, 'pearson_p_raw': pear_p,
            'spearman_r_raw': spear_r, 'spearman_p_raw': spear_p,
            'timescale_mean_ms_qc': sub.timescale_ms.mean(),
            'timescale_median_ms_qc': sub.timescale_ms.median(),
            'age_min_qc': sub.age.min(), 'age_max_qc': sub.age.max(),
        })
    res = pd.DataFrame(rows)
    res['age_q_adjusted_model'] = fdr(res['age_p_adjusted_model'])
    res['pearson_q_raw'] = fdr(res['pearson_p_raw'])
    res['spearman_q_raw'] = fdr(res['spearman_p_raw'])
    res.to_csv(OUT / 'ComBat_timescale_age_association_DK68_QC_nonboundary_fitR2gt0.csv', index=False)
    return res


def global_age_models(df):
    subj = df.groupby(['subject_id','site','age','sex']).agg(
        global_timescale_ms_mean_qc=('timescale_ms','mean'),
        global_timescale_ms_median_qc=('timescale_ms','median'),
        n_regions_qc=('region','count'),
    ).reset_index()
    rows = []
    for outcome in ['global_timescale_ms_mean_qc','global_timescale_ms_median_qc']:
        sub = subj[np.isfinite(subj[outcome]) & np.isfinite(subj.age)].copy()
        pear_r, pear_p = stats.pearsonr(sub.age, sub[outcome])
        spear_r, spear_p = stats.spearmanr(sub.age, sub[outcome])
        m = smf.ols(f'{outcome} ~ age + C(sex) + C(site)', data=sub).fit(cov_type='HC3')
        rows.append({
            'outcome': outcome, 'n_subjects': len(sub),
            'age_beta_ms_per_year_adjusted': m.params.get('age', np.nan),
            'age_se': m.bse.get('age', np.nan), 'age_t': m.tvalues.get('age', np.nan),
            'age_p_adjusted_model': m.pvalues.get('age', np.nan), 'model_r2': m.rsquared,
            'pearson_r_raw': pear_r, 'pearson_p_raw': pear_p,
            'spearman_r_raw': spear_r, 'spearman_p_raw': spear_p,
        })
        for site, ss in sub.groupby('site'):
            pr, pp = stats.pearsonr(ss.age, ss[outcome]) if len(ss)>5 else (np.nan,np.nan)
            sr, sp = stats.spearmanr(ss.age, ss[outcome]) if len(ss)>5 else (np.nan,np.nan)
            rows.append({'outcome': outcome + f'__{site}', 'n_subjects': len(ss),
                         'age_beta_ms_per_year_adjusted': np.polyfit(ss.age, ss[outcome], 1)[0] if len(ss)>1 else np.nan,
                         'age_se': np.nan, 'age_t': np.nan, 'age_p_adjusted_model': pp,
                         'model_r2': np.nan, 'pearson_r_raw': pr, 'pearson_p_raw': pp,
                         'spearman_r_raw': sr, 'spearman_p_raw': sp})
    out = pd.DataFrame(rows)
    out['age_q_adjusted_model'] = fdr(out.age_p_adjusted_model)
    subj.to_csv(OUT / 'ComBat_global_timescale_subject_level_values_QC.csv', index=False)
    out.to_csv(OUT / 'ComBat_global_timescale_age_association_subject_level_QC.csv', index=False)
    return subj, out


def plot_global_clean(subj):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = {'CAM_CAN':'#4C78A8','OMEGA':'#F58518','PANS':'#54A24B'}
    for site, ss in subj.groupby('site'):
        ax.scatter(ss.age, ss.global_timescale_ms_mean_qc, s=24, alpha=0.65, color=colors.get(site), edgecolors='none', label=site)
    x = subj.age.to_numpy(float); y = subj.global_timescale_ms_mean_qc.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum()>2:
        b,a = np.polyfit(x[ok], y[ok], 1); xx = np.linspace(x[ok].min(), x[ok].max(), 200); ax.plot(xx, a+b*xx, color='black', lw=2)
    ax.set_xlabel('Age (years)'); ax.set_ylabel('QC global mean timescale (ms)')
    ax.grid(color='#dddddd', linewidth=0.7); ax.legend(frameon=False, title='site')
    fig.savefig(OUT / 'ComBat_global_timescale_age_scatter_QC.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_beta_bar(res):
    d = res.sort_values('age_beta_ms_per_year_adjusted')
    fig, ax = plt.subplots(figsize=(8.4, 12.5))
    colors = np.where(d.age_q_adjusted_model < 0.05, '#D55E00', '#4C78A8')
    ax.barh(d.region_name, d.age_beta_ms_per_year_adjusted, color=colors, alpha=0.82)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Age slope after QC (ms/year)'); ax.set_ylabel('DK region')
    ax.grid(axis='x', color='#dddddd', linewidth=0.7)
    fig.savefig(OUT / 'ComBat_timescale_age_beta_DK68_barplot_QC.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    df = pd.read_csv(IN_CSV)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    before = len(df)
    qc = df.query(QC_QUERY).copy()
    (OUT / 'QC_filter.txt').write_text(
        f'Filter: {QC_QUERY}\nRows before: {before}\nRows after: {len(qc)}\nSubjects after: {qc.subject_id.nunique()}\n', encoding='utf-8')
    res = region_age_models(qc)
    subj, glob = global_age_models(qc)
    plot_global_clean(subj)
    plot_beta_bar(res)
    render_clean_brain(res.sort_values('region').age_beta_ms_per_year_adjusted.to_numpy(), OUT / 'ComBat_timescale_age_beta_DK68_brainmap_QC_clean.png', diverging=True)
    render_clean_brain(res.sort_values('region').timescale_mean_ms_qc.to_numpy(), OUT / 'ComBat_timescale_DK68_mean_QC_clean.png', diverging=False)
    print('Wrote QC age association outputs to', OUT)
    print(glob.head(10).to_string(index=False))
    print('FDR<0.05 regions:', int((res.age_q_adjusted_model < 0.05).sum()))

if __name__ == '__main__':
    main()
