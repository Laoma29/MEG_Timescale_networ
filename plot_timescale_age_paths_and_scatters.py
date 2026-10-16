import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path(r'C:\Users\laoma\.codex\visualizations\2026\09\14\01a09fbb-0280-7fb2-b300-2e243f47d187\timescale_power_fooof')
AGE_DIR = BASE / 'age_association'
OUT = BASE / 'age_region_path_and_scatter_python'
SCATTER_DIR = OUT / 'per_region_scatter'
OUT.mkdir(exist_ok=True)
SCATTER_DIR.mkdir(exist_ok=True)

mpl.rcParams.update({
    'font.family':'sans-serif',
    'font.sans-serif':['Arial','Helvetica','DejaVu Sans','sans-serif'],
    'pdf.fonttype':42,
    'svg.fonttype':'none',
    'axes.spines.top':False,
    'axes.spines.right':False,
    'axes.linewidth':1.0,
    'xtick.major.width':1.0,
    'ytick.major.width':1.0,
    'font.size':8,
})

def savefig(fig, stem, dpi=600):
    fig.savefig(str(stem)+'.png', dpi=300, bbox_inches='tight')
    fig.savefig(str(stem)+'.pdf', bbox_inches='tight')
    fig.savefig(str(stem)+'.svg', bbox_inches='tight')

long = pd.read_csv(BASE / 'timescale_ComBat_model_ready_long.csv')
stats_df = pd.read_csv(AGE_DIR / 'ComBat_timescale_age_association_DK68_timescale_ms.csv')
# Merge region names and adjusted age beta.
df = long.merge(stats_df[['region','region_name','age_beta_ms_per_year_adjusted','age_q_adjusted_model','spearman_r_raw','spearman_p_raw']], on='region', how='left')
# Robust winsorization per region for display only.
def prep_region(g):
    y = g['timescale_ms'].astype(float).to_numpy()
    lo, hi = np.nanpercentile(y, [2, 98])
    g = g.copy()
    g['timescale_ms_winsor'] = np.clip(y, lo, hi)
    g['timescale_s_winsor'] = g['timescale_ms_winsor'] / 1000.0
    g['timescale_s_demeaned'] = g['timescale_s_winsor'] - np.nanmean(g['timescale_s_winsor'])
    return g

df = df.groupby('region', group_keys=False).apply(prep_region)

# Use fixed color gradient like example, indexed by region number.
cmap = mpl.cm.get_cmap('plasma')
regions = sorted(df['region'].unique())
region_color = {r:cmap((i)/(len(regions)-1)) for i,r in enumerate(regions)}

# Region age path: linear fit for each region on demeaned seconds.
age_grid = np.linspace(df.age.min(), df.age.max(), 160)
path_rows=[]
fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=220)
for r in regions:
    g = df[df.region == r]
    x = g['age'].to_numpy(float)
    y = g['timescale_s_demeaned'].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        continue
    slope, intercept, rv, pv, se = stats.linregress(x[m], y[m])
    yhat = intercept + slope * age_grid
    ax.plot(age_grid, yhat, color=region_color[r], alpha=0.23, lw=0.9)
    path_rows.append({'region':r,'region_name':g.region_name.iloc[0], 'display_slope_s_per_year':slope, 'display_intercept':intercept, 'pearson_r_display':rv, 'p_display':pv})
# Mean fitted trajectory across regions.
paths = []
for r in regions:
    rr = [x for x in path_rows if x['region']==r]
    if not rr: continue
    paths.append(rr[0]['display_intercept'] + rr[0]['display_slope_s_per_year'] * age_grid)
paths = np.vstack(paths)
ax.plot(age_grid, np.nanmean(paths, axis=0), color='0.35', lw=2.8, zorder=5)
ax.axhline(0, color='0.88', lw=1)
ax.set_xlabel('Age (years)', fontsize=10)
ax.set_ylabel('Demeaned timescale (s)', fontsize=10)
ax.set_title('Regional timescale-age trajectories', loc='left', fontsize=11)
ax.tick_params(labelsize=9, length=4)
# Compact horizontal colorbar for region index.
norm = mpl.colors.Normalize(vmin=min(regions), vmax=max(regions))
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.08, pad=0.24)
cbar.set_label('DK region index', fontsize=8)
cbar.ax.tick_params(labelsize=7, length=2)
fig.tight_layout()
savefig(fig, OUT / 'DK68_timescale_age_region_path_demeaned')
plt.close(fig)
pd.DataFrame(path_rows).to_csv(OUT / 'DK68_timescale_age_region_path_fit_display.csv', index=False)

# Alternative path colored by age beta direction/significance, useful for manuscript.
fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=220)
for r in regions:
    g = df[df.region == r]
    x = g['age'].to_numpy(float); y = g['timescale_s_demeaned'].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10: continue
    slope, intercept, rv, pv, se = stats.linregress(x[m], y[m])
    beta = g['age_beta_ms_per_year_adjusted'].iloc[0]
    q = g['age_q_adjusted_model'].iloc[0]
    col = '#c7353d' if beta > 0 else '#3156c9'
    alpha = 0.34 if q < 0.05 else 0.12
    lw = 1.1 if q < 0.05 else 0.7
    ax.plot(age_grid, intercept + slope*age_grid, color=col, alpha=alpha, lw=lw)
ax.axhline(0, color='0.88', lw=1)
ax.set_xlabel('Age (years)', fontsize=10)
ax.set_ylabel('Demeaned timescale (s)', fontsize=10)
ax.set_title('Regional trajectories by age-effect direction', loc='left', fontsize=11)
ax.tick_params(labelsize=9, length=4)
handles=[plt.Line2D([0],[0],color='#c7353d',lw=2,label='positive beta'), plt.Line2D([0],[0],color='#3156c9',lw=2,label='negative beta')]
ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper left')
fig.tight_layout()
savefig(fig, OUT / 'DK68_timescale_age_region_path_by_beta')
plt.close(fig)

# Individual per-region scatter plots using attachment style.
scatter_rows=[]
for r in regions:
    g = df[df.region == r].copy()
    x = g['age'].to_numpy(float)
    # Display in seconds but centered for better axes, same spirit as reference.
    y = g['timescale_s_demeaned'].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        continue
    slope, intercept, rv, pv, se = stats.linregress(x[m], y[m])
    rho, sp = stats.spearmanr(x[m], y[m])
    yhat = intercept + slope * age_grid
    rn = g['region_name'].iloc[0]
    fig, ax = plt.subplots(figsize=(2.35, 2.2), dpi=240)
    ax.scatter(x[m], y[m], s=8, color='#b8b8b8', edgecolors='none', alpha=0.72, rasterized=True)
    ax.plot(age_grid, yhat, color='#8d8d8d', lw=2.7)
    ax.text(0.96, 0.92, r'$r_s$ = ' + f'{rho:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=9)
    ax.set_xlabel('Age (years)', fontsize=9)
    ax.set_ylabel('Demeaned timescale (s)', fontsize=9)
    short = rn.replace('lh_', 'L ').replace('rh_', 'R ').replace('_', ' ')
    ax.set_title(short, loc='left', fontsize=9)
    ax.tick_params(labelsize=8, length=4)
    # symmetrical y-limit around robust percentile.
    ylim = np.nanpercentile(np.abs(y[m]), 98)
    ylim = max(ylim, 0.05)
    ax.set_ylim(-ylim*1.08, ylim*1.08)
    fig.tight_layout(pad=0.7)
    stem = SCATTER_DIR / f'region_{int(r):02d}_{rn}_timescale_age_scatter'
    savefig(fig, stem)
    plt.close(fig)
    scatter_rows.append({'region':r,'region_name':rn,'n':int(m.sum()),'slope_s_per_year_display':slope,'pearson_r_display':rv,'pearson_p_display':pv,'spearman_r_display':rho,'spearman_p_display':sp,'adjusted_beta_ms_per_year':g.age_beta_ms_per_year_adjusted.iloc[0],'adjusted_q':g.age_q_adjusted_model.iloc[0], 'png':str(stem)+'.png'})
scatter_index = pd.DataFrame(scatter_rows)
scatter_index.to_csv(OUT / 'per_region_scatter_index.csv', index=False)

# Make a compact panel for strongest positive and negative adjusted beta regions.
top_pos = stats_df.sort_values('age_beta_ms_per_year_adjusted', ascending=False).head(3)['region'].tolist()
top_neg = stats_df.sort_values('age_beta_ms_per_year_adjusted', ascending=True).head(3)['region'].tolist()
sel = top_pos + top_neg
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), dpi=220, sharex=True)
for ax, r in zip(axes.ravel(), sel):
    g = df[df.region == r]
    x = g.age.to_numpy(float); y = g.timescale_s_demeaned.to_numpy(float)
    m=np.isfinite(x)&np.isfinite(y)
    slope, intercept, rv, pv, se=stats.linregress(x[m], y[m])
    rho, sp=stats.spearmanr(x[m], y[m])
    ax.scatter(x[m], y[m], s=7, color='#b8b8b8', edgecolors='none', alpha=.68, rasterized=True)
    ax.plot(age_grid, intercept+slope*age_grid, color='#8d8d8d', lw=2.5)
    ax.text(.96,.90,r'$r_s$ = '+f'{rho:.2f}',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    ax.set_title(g.region_name.iloc[0].replace('lh_','L ').replace('rh_','R ').replace('_',' '),loc='left',fontsize=8)
    ax.tick_params(labelsize=7,length=3)
    ylim=np.nanpercentile(np.abs(y[m]),98); ylim=max(ylim,.05); ax.set_ylim(-ylim*1.08,ylim*1.08)
for ax in axes[-1,:]: ax.set_xlabel('Age (years)', fontsize=8)
for ax in axes[:,0]: ax.set_ylabel('Demeaned timescale (s)', fontsize=8)
fig.tight_layout()
savefig(fig, OUT / 'top_positive_negative_regions_scatter_panel')
plt.close(fig)

# Save a lightweight summary for final reporting.
summary = {
    'input': str(BASE / 'timescale_ComBat_model_ready_long.csv'),
    'n_rows': int(len(df)),
    'n_subjects': int(df.subject_id.nunique()),
    'n_regions': int(df.region.nunique()),
    'out_dir': str(OUT),
    'scatter_dir': str(SCATTER_DIR),
    'top_positive_regions': stats_df.sort_values('age_beta_ms_per_year_adjusted', ascending=False).head(5)[['region','region_name','age_beta_ms_per_year_adjusted','age_q_adjusted_model']].to_dict('records'),
    'top_negative_regions': stats_df.sort_values('age_beta_ms_per_year_adjusted', ascending=True).head(5)[['region','region_name','age_beta_ms_per_year_adjusted','age_q_adjusted_model']].to_dict('records'),
}
(OUT / 'timescale_age_path_scatter_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('DONE', OUT)
print('scatter plots', len(scatter_rows), SCATTER_DIR)
print('top positive')
print(pd.DataFrame(summary['top_positive_regions']).to_string(index=False))
print('top negative')
print(pd.DataFrame(summary['top_negative_regions']).to_string(index=False))
