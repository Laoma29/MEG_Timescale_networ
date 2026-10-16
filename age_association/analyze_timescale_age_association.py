# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(r'C:\Users\laoma\.codex\visualizations\2026\09\14\01a09fbb-0280-7fb2-b300-2e243f47d187\timescale_power_fooof')
IN_CSV = BASE / 'timescale_ComBat_model_ready_long.csv'
DK_SUMMARY = BASE / 'dk_brain_maps' / 'ComBat_timescale_DK68_All_68x1.csv'
OUT = BASE / 'age_association'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})


def zscore_by_region(df, value_col):
    return df.groupby('region')[value_col].transform(lambda x: (x - x.mean()) / x.std(ddof=0))


def fdr(p):
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan, dtype=float)
    if ok.sum():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q


def region_models(df, value_col='timescale_ms'):
    region_names = pd.read_csv(DK_SUMMARY)[['region','region_name']]
    rows = []
    for region, sub in df.groupby('region'):
        sub = sub[np.isfinite(sub[value_col]) & np.isfinite(sub['age'])].copy()
        if len(sub) < 30:
            continue
        pear_r, pear_p = stats.pearsonr(sub['age'], sub[value_col])
        spear_r, spear_p = stats.spearmanr(sub['age'], sub[value_col], nan_policy='omit')
        # Robust-ish primary regression: adjust for sex and site as categorical covariates.
        try:
            m = smf.ols(f'{value_col} ~ age + C(sex) + C(site)', data=sub).fit(cov_type='HC3')
            beta = m.params.get('age', np.nan)
            se = m.bse.get('age', np.nan)
            t = m.tvalues.get('age', np.nan)
            p = m.pvalues.get('age', np.nan)
            ci = m.conf_int().loc['age'].to_numpy() if 'age' in m.params.index else [np.nan, np.nan]
            r2 = m.rsquared
        except Exception:
            beta = se = t = p = r2 = np.nan
            ci = [np.nan, np.nan]
        rows.append({
            'region': int(region), 'n': len(sub),
            'age_beta_ms_per_year_adjusted': beta,
            'age_beta_ci_low': ci[0], 'age_beta_ci_high': ci[1],
            'age_se': se, 'age_t': t, 'age_p_adjusted_model': p,
            'model_r2': r2,
            'pearson_r_raw': pear_r, 'pearson_p_raw': pear_p,
            'spearman_r_raw': spear_r, 'spearman_p_raw': spear_p,
            'timescale_mean_ms': sub[value_col].mean(),
            'timescale_median_ms': sub[value_col].median(),
            'age_min': sub['age'].min(), 'age_max': sub['age'].max(),
        })
    res = pd.DataFrame(rows).merge(region_names, on='region', how='left')
    res['age_q_adjusted_model'] = fdr(res['age_p_adjusted_model'])
    res['pearson_q_raw'] = fdr(res['pearson_p_raw'])
    res['spearman_q_raw'] = fdr(res['spearman_p_raw'])
    res = res[['region','region_name','n','age_beta_ms_per_year_adjusted','age_beta_ci_low','age_beta_ci_high',
               'age_se','age_t','age_p_adjusted_model','age_q_adjusted_model','model_r2',
               'pearson_r_raw','pearson_p_raw','pearson_q_raw','spearman_r_raw','spearman_p_raw','spearman_q_raw',
               'timescale_mean_ms','timescale_median_ms','age_min','age_max']]
    res.to_csv(OUT / f'ComBat_timescale_age_association_DK68_{value_col}.csv', index=False)
    return res


def subject_global_models(df):
    subj = df.groupby(['subject_id','site','age','sex']).agg(
        global_timescale_ms_mean=('timescale_ms','mean'),
        global_timescale_ms_median=('timescale_ms','median'),
        global_knee_frequency_hz_mean=('knee_frequency_hz','mean'),
        mean_fit_r2=('fit_r2','mean'),
        lower_bound_fraction=('knee_at_lower_grid_bound','mean'),
        upper_bound_fraction=('knee_at_upper_grid_bound','mean'),
    ).reset_index()
    rows = []
    for value_col in ['global_timescale_ms_mean','global_timescale_ms_median']:
        pear_r, pear_p = stats.pearsonr(subj['age'], subj[value_col])
        spear_r, spear_p = stats.spearmanr(subj['age'], subj[value_col])
        m = smf.ols(f'{value_col} ~ age + C(sex) + C(site)', data=subj).fit(cov_type='HC3')
        rows.append({
            'outcome': value_col,
            'n_subjects': len(subj),
            'age_beta_ms_per_year_adjusted': m.params.get('age', np.nan),
            'age_se': m.bse.get('age', np.nan),
            'age_t': m.tvalues.get('age', np.nan),
            'age_p_adjusted_model': m.pvalues.get('age', np.nan),
            'model_r2': m.rsquared,
            'pearson_r_raw': pear_r,
            'pearson_p_raw': pear_p,
            'spearman_r_raw': spear_r,
            'spearman_p_raw': spear_p,
        })
        for site, ss in subj.groupby('site'):
            if len(ss) > 5:
                pr, pp = stats.pearsonr(ss['age'], ss[value_col])
                sr, sp = stats.spearmanr(ss['age'], ss[value_col])
            else:
                pr=pp=sr=sp=np.nan
            rows.append({
                'outcome': value_col + f'__{site}',
                'n_subjects': len(ss),
                'age_beta_ms_per_year_adjusted': np.polyfit(ss['age'], ss[value_col], 1)[0] if len(ss)>1 else np.nan,
                'age_se': np.nan, 'age_t': np.nan, 'age_p_adjusted_model': pp,
                'model_r2': np.nan, 'pearson_r_raw': pr, 'pearson_p_raw': pp,
                'spearman_r_raw': sr, 'spearman_p_raw': sp,
            })
    out = pd.DataFrame(rows)
    out['age_q_adjusted_model'] = fdr(out['age_p_adjusted_model'])
    out.to_csv(OUT / 'ComBat_global_timescale_age_association_subject_level.csv', index=False)
    subj.to_csv(OUT / 'ComBat_global_timescale_subject_level_values.csv', index=False)
    return subj, out


def plot_global(subj):
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    colors = {'CAM_CAN': '#4C78A8', 'OMEGA': '#F58518', 'PANS': '#54A24B'}
    for site, ss in subj.groupby('site'):
        ax.scatter(ss['age'], ss['global_timescale_ms_mean'], s=28, alpha=0.65,
                   color=colors.get(site, None), edgecolors='none', label=site)
    x = subj['age'].to_numpy(dtype=float)
    y = subj['global_timescale_ms_mean'].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 2:
        b, a = np.polyfit(x[ok], y[ok], 1)
        xx = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 200)
        ax.plot(xx, a + b * xx, color='black', lw=2)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Global mean timescale (ms)')
    ax.set_title('ComBat DK68 global timescale vs age', fontweight='bold')
    ax.grid(color='#dddddd', linewidth=0.7)
    ax.legend(frameon=False, title='site')
    fig.savefig(OUT / 'ComBat_global_timescale_age_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_region_beta(res):
    d = res.sort_values('age_beta_ms_per_year_adjusted')
    fig, ax = plt.subplots(figsize=(8.2, 12.5))
    colors = np.where(d['age_q_adjusted_model'] < 0.05, '#D55E00', '#4C78A8')
    ax.barh(d['region_name'], d['age_beta_ms_per_year_adjusted'], color=colors, alpha=0.82)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Age slope, adjusted model (ms/year)')
    ax.set_ylabel('DK region')
    ax.set_title('Age association with ComBat timescale across DK68', fontweight='bold')
    ax.grid(axis='x', color='#dddddd', linewidth=0.7)
    fig.savefig(OUT / 'ComBat_timescale_age_beta_DK68_barplot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_brain_age_beta(res):
    # Reuse DK plotting script assets by importing its functions if available.
    import sys
    dk_dir = BASE / 'dk_brain_maps'
    sys.path.insert(0, str(dk_dir))
    import plot_combat_timescale_dk68 as dkplot
    tmp = res[['region','region_name','age_beta_ms_per_year_adjusted']].copy()
    tmp['site_group'] = 'Age_beta'
    tmp['timescale_ms_mean'] = tmp['age_beta_ms_per_year_adjusted']
    # Monkey-patch output in the DK renderer: it expects summary columns and writes a fixed filename pattern.
    dkplot.OUT = OUT
    dkplot.render_map(tmp, 'Age_beta', metric='timescale_ms_mean')
    src = OUT / 'ComBat_timescale_DK68_Age_beta_brainmap.png'
    dst = OUT / 'ComBat_timescale_age_beta_DK68_brainmap.png'
    if src.exists():
        src.replace(dst)


def main():
    df = pd.read_csv(IN_CSV)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df[np.isfinite(df['age'])].copy()
    res = region_models(df, 'timescale_ms')
    subj, glob = subject_global_models(df)
    plot_global(subj)
    plot_region_beta(res)
    try:
        plot_brain_age_beta(res)
    except Exception as e:
        (OUT / 'brain_age_beta_plot_error.txt').write_text(str(e), encoding='utf-8')
    print('Wrote age association results to', OUT)
    print(glob.head(10).to_string(index=False))

if __name__ == '__main__':
    main()

