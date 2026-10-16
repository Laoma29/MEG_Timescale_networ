# -*- coding: utf-8 -*-
"""Fast timescale computation from PSD + FOOOF exponent.

Reference: Gao et al., eLife 2020. Timescale tau = 1/(2*pi*f_k).
Here f_k is estimated by fitting the aperiodic knee model with exponent fixed to
existing FOOOF/specparam exponent:
    log10(P) = offset - log10(f_k**exponent + f**exponent)
For each subject x region, f_k is selected by grid search and offset is solved
analytically for each candidate f_k.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import h5py
from scipy.io import savemat

MAT_PATH = Path(r'E:\功率谱常模\power_normative_input_struct_RAW_and_ComBat_with_ID_site_age_sex.mat')
FEATURE_INFO_CSV = Path(r'E:\功率谱常模\power_normative_input_feature_info.csv')
SUBJECT_INFO_CSV = Path(r'E:\功率谱常模\power_normative_input_subject_info.csv')
OUT = Path(r'C:\Users\laoma\.codex\visualizations\2026\09\14\01a09fbb-0280-7fb2-b300-2e243f47d187\timescale_power_fooof')
OUT.mkdir(parents=True, exist_ok=True)
FREQ_MIN, FREQ_MAX = 2.0, 40.0
N_REGIONS = 68
FK_GRID = np.geomspace(0.05, 100.0, 320)  # Hz; broader range to flag knees outside the 2-40 Hz fit band


def subject_info_model_ready(f, root):
    subj = pd.read_csv(SUBJECT_INFO_CSV)
    valid = np.array(f[root]['valid_subject']).astype(bool).ravel()
    if valid.size == len(subj):
        subj = subj.loc[valid].reset_index(drop=True)
    else:
        subj = subj.iloc[:int(f[root]['age'].shape[-1])].copy().reset_index(drop=True)
    return subj


def raw_region_values(f):
    feat = pd.read_csv(FEATURE_INFO_CSV)
    Y = np.array(f['NormativeData_raw']['Y_model_ready']).T
    subj = subject_info_model_ready(f, 'NormativeData_raw')
    freq = np.array(f['NormativeData_raw']['freq_grid']).ravel()
    n_sub = Y.shape[0]
    abs_power = np.full((freq.size, N_REGIONS, n_sub), np.nan, dtype=np.float64)
    exponent = np.full((N_REGIONS, n_sub), np.nan, dtype=np.float64)
    offset = np.full((N_REGIONS, n_sub), np.nan, dtype=np.float64)
    for idx, row in feat.iterrows():
        r = int(row['region']) - 1
        ft = row['feature_type']
        if ft == 'absolute_power':
            hz = float(row['frequency_hz'])
            fi = np.where(np.isclose(freq, hz))[0]
            if fi.size:
                abs_power[fi[0], r, :] = Y[:, idx]
        elif ft == 'aperiodic_exponent':
            exponent[r, :] = Y[:, idx]
        elif ft == 'aperiodic_offset':
            offset[r, :] = Y[:, idx]
    return freq, abs_power, exponent, offset, subj


def combat_region_values(f):
    g = f['NormativeData_combat']['region_values']
    return (np.array(g['freq_grid']).ravel(), np.array(g['abs_power']),
            np.array(g['aperiodic_exponent']), np.array(g['aperiodic_offset']),
            subject_info_model_ready(f, 'NormativeData_combat'))


def fit_fixed_exponent_matrix(freq, logp_freq_x_sub, exponent_sub):
    """Fit all subjects for one region. Returns dict arrays length n_subject."""
    y = logp_freq_x_sub.astype(float)  # n_freq x n_sub
    n_freq, n_sub = y.shape
    expv = exponent_sub.astype(float).copy()
    bad_exp = ~np.isfinite(expv) | (expv <= 0.05) | (expv > 8)
    if np.any(bad_exp):
        # slope fallback per subject
        x = np.log10(freq)
        for s in np.where(bad_exp)[0]:
            yy = y[:, s]
            m = np.isfinite(yy)
            if m.sum() >= 8:
                expv[s] = np.clip(-np.polyfit(x[m], yy[m], 1)[0], 0.2, 5.0)
            else:
                expv[s] = np.nan
    expv = np.clip(expv, 0.05, 8.0)

    best_sse = np.full(n_sub, np.inf)
    best_fk = np.full(n_sub, np.nan)
    best_offset = np.full(n_sub, np.nan)
    best_rmse = np.full(n_sub, np.nan)
    valid_counts = np.sum(np.isfinite(y), axis=0)

    # Loop over candidate knee frequencies; solve offset analytically.
    for fk in FK_GRID:
        # term[freq, subject] = log10(fk^exp + f^exp)
        term = np.log10(np.power(fk, expv)[None, :] + np.power(freq[:, None], expv[None, :]))
        z = y + term
        offset = np.nanmean(z, axis=0)
        pred = offset[None, :] - term
        resid = pred - y
        sse = np.nansum(resid * resid, axis=0)
        improve = sse < best_sse
        best_sse[improve] = sse[improve]
        best_fk[improve] = fk
        best_offset[improve] = offset[improve]

    ymean = np.nanmean(y, axis=0)
    sst = np.nansum((y - ymean[None, :]) ** 2, axis=0)
    r2 = 1.0 - best_sse / sst
    best_rmse = np.sqrt(best_sse / np.maximum(valid_counts, 1))
    invalid = (valid_counts < 8) | ~np.isfinite(expv) | (sst <= 0)
    best_fk[invalid] = np.nan
    best_offset[invalid] = np.nan
    expv[invalid] = np.nan
    r2[invalid] = np.nan
    best_rmse[invalid] = np.nan
    tau_s = 1.0 / (2.0 * np.pi * best_fk)
    tau_s[~np.isfinite(best_fk) | (best_fk <= 0)] = np.nan
    knee_param = np.power(best_fk, expv)
    return dict(fk=best_fk, tau_s=tau_s, knee_param=knee_param, offset=best_offset,
                exponent=expv, r2=r2, rmse=best_rmse)


def compute_dataset(name, freq, abs_power, exponent_init, offset_init, subj):
    mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
    fit_freq = freq[mask]
    n_freq, n_regions, n_subjects = abs_power.shape
    tau = np.full((n_regions, n_subjects), np.nan)
    fk = np.full((n_regions, n_subjects), np.nan)
    knee_param = np.full((n_regions, n_subjects), np.nan)
    fit_offset = np.full((n_regions, n_subjects), np.nan)
    fit_exp = np.full((n_regions, n_subjects), np.nan)
    fit_r2 = np.full((n_regions, n_subjects), np.nan)
    fit_rmse = np.full((n_regions, n_subjects), np.nan)
    records = []
    t0 = time.time()
    for r in range(n_regions):
        print(f'{name}: region {r+1}/{n_regions}', flush=True)
        res = fit_fixed_exponent_matrix(fit_freq, abs_power[mask, r, :], exponent_init[r, :])
        fk[r, :] = res['fk']; tau[r, :] = res['tau_s']; knee_param[r, :] = res['knee_param']
        fit_offset[r, :] = res['offset']; fit_exp[r, :] = res['exponent']; fit_r2[r, :] = res['r2']; fit_rmse[r, :] = res['rmse']
    for s in range(n_subjects):
        sid = subj.loc[s, 'subject_id'] if 'subject_id' in subj.columns and s < len(subj) else f'sub-{s+1:04d}'
        site = subj.loc[s, 'site'] if 'site' in subj.columns and s < len(subj) else ''
        age = subj.loc[s, 'age'] if 'age' in subj.columns and s < len(subj) else np.nan
        sex = subj.loc[s, 'sex'] if 'sex' in subj.columns and s < len(subj) else ''
        for r in range(n_regions):
            records.append({
                'dataset_version': name, 'subject_index': s+1, 'subject_id': sid,
                'site': site, 'age': age, 'sex': sex, 'region': r+1,
                'knee_frequency_hz': fk[r, s], 'timescale_s': tau[r, s],
                'timescale_ms': tau[r, s] * 1000 if np.isfinite(tau[r, s]) else np.nan,
                'knee_parameter': knee_param[r, s],
                'aperiodic_exponent_fixed': fit_exp[r, s],
                'aperiodic_offset_fit': fit_offset[r, s],
                'aperiodic_offset_initial': offset_init[r, s],
                'fit_r2': fit_r2[r, s], 'fit_rmse_log10': fit_rmse[r, s],
                'fit_freq_min_hz': FREQ_MIN, 'fit_freq_max_hz': FREQ_MAX,
                'fk_grid_min_hz': float(FK_GRID.min()), 'fk_grid_max_hz': float(FK_GRID.max()),
                'knee_at_lower_grid_bound': bool(np.isfinite(fk[r, s]) and fk[r, s] <= FK_GRID[0] * 1.000001),
                'knee_at_upper_grid_bound': bool(np.isfinite(fk[r, s]) and fk[r, s] >= FK_GRID[-1] / 1.000001),
                'method': 'fixed_fooof_exponent_gridsearch_knee',
            })
    df = pd.DataFrame.from_records(records)
    df.to_csv(OUT / f'timescale_{name}_long.csv', index=False)
    df.groupby('region').agg(
        n=('timescale_ms','count'), timescale_ms_mean=('timescale_ms','mean'),
        timescale_ms_sd=('timescale_ms','std'), timescale_ms_median=('timescale_ms','median'),
        knee_frequency_hz_mean=('knee_frequency_hz','mean'), knee_frequency_hz_median=('knee_frequency_hz','median'),
        fit_r2_mean=('fit_r2','mean'), fit_rmse_log10_mean=('fit_rmse_log10','mean'),
    ).reset_index().to_csv(OUT / f'timescale_{name}_region_summary.csv', index=False)
    df.groupby(['subject_index','subject_id','site','age','sex']).agg(
        timescale_ms_mean=('timescale_ms','mean'), timescale_ms_median=('timescale_ms','median'),
        knee_frequency_hz_mean=('knee_frequency_hz','mean'), fit_r2_mean=('fit_r2','mean'),
    ).reset_index().to_csv(OUT / f'timescale_{name}_subject_summary.csv', index=False)
    savemat(OUT / f'timescale_{name}.mat', {
        'timescale_s_region_x_subject': tau,
        'timescale_ms_region_x_subject': tau * 1000,
        'knee_frequency_hz_region_x_subject': fk,
        'knee_parameter_region_x_subject': knee_param,
        'aperiodic_exponent_fixed_region_x_subject': fit_exp,
        'aperiodic_offset_fit_region_x_subject': fit_offset,
        'fit_r2_region_x_subject': fit_r2,
        'fit_rmse_log10_region_x_subject': fit_rmse,
        'fit_freq_range_hz': np.array([FREQ_MIN, FREQ_MAX]),
        'fk_grid_hz': FK_GRID,
    }, do_compression=True)
    print(f'{name}: finished in {time.time()-t0:.1f}s', flush=True)


def main():
    meta = dict(
        source_mat=str(MAT_PATH),
        reference='Gao et al., eLife 2020;9:e61277. tau = 1/(2*pi*f_k).',
        method='Aperiodic knee model with FOOOF exponent fixed; grid-search knee frequency; analytic offset.',
        model='log10(P)=offset-log10(f_k**exponent + f**exponent)',
        fit_frequency_range_hz=[FREQ_MIN, FREQ_MAX],
        fk_grid_hz=[float(FK_GRID.min()), float(FK_GRID.max()), int(FK_GRID.size)],
        note='The provided FOOOF results include fixed aperiodic exponent/offset but no knee parameter, so f_k is re-estimated from PSD.',
    )
    (OUT / 'timescale_method_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    with h5py.File(MAT_PATH, 'r') as f:
        freq, abs_power, exp0, off0, subj = raw_region_values(f)
        compute_dataset('RAW_model_ready', freq, abs_power, exp0, off0, subj)
        freq, abs_power, exp0, off0, subj = combat_region_values(f)
        compute_dataset('ComBat_model_ready', freq, abs_power, exp0, off0, subj)

if __name__ == '__main__':
    main()

