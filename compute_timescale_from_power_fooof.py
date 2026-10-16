# -*- coding: utf-8 -*-
"""Compute neuronal timescale from PSD knee frequency following Gao et al., eLife 2020.

Timescale definition:
  tau = 1 / (2*pi*f_k)
with f_k estimated from the aperiodic knee model:
  log10(P) = offset - log10(knee + f**exponent)
where f_k = knee ** (1/exponent).
"""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
import h5py
from scipy.optimize import least_squares
from scipy.io import savemat

warnings.filterwarnings('ignore')

MAT_PATH = Path(r'E:\功率谱常模\power_normative_input_struct_RAW_and_ComBat_with_ID_site_age_sex.mat')
FEATURE_INFO_CSV = Path(r'E:\功率谱常模\power_normative_input_feature_info.csv')
SUBJECT_INFO_CSV = Path(r'E:\功率谱常模\power_normative_input_subject_info.csv')
OUT = Path(r'C:\Users\laoma\.codex\visualizations\2026\09\14\01a09fbb-0280-7fb2-b300-2e243f47d187\timescale_power_fooof')
OUT.mkdir(parents=True, exist_ok=True)

FREQ_MIN = 2.0
FREQ_MAX = 40.0
N_REGIONS = 68


def knee_model_log10(freq, offset, log_knee, exponent):
    knee = np.exp(log_knee)
    return offset - np.log10(knee + np.power(freq, exponent))


def fit_one(freq, logp, exponent0=np.nan, offset0=np.nan):
    m = np.isfinite(freq) & np.isfinite(logp)
    f = freq[m].astype(float)
    y = logp[m].astype(float)
    if f.size < 12 or np.nanstd(y) < 1e-8:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False

    if not np.isfinite(offset0):
        offset0 = float(np.nanmax(y) + np.log10(np.nanmax(f) + 1.0))
    if not np.isfinite(exponent0) or exponent0 <= 0.05 or exponent0 > 6.0:
        # Slope-based fallback initialization.
        coef = np.polyfit(np.log10(f), y, 1)
        exponent0 = float(np.clip(-coef[0], 0.2, 4.5))
    exponent0 = float(np.clip(exponent0, 0.1, 5.0))
    # Initialize the knee near 5 Hz in the FOOOF/specparam knee parameterization.
    log_knee0 = float(np.log(np.power(5.0, exponent0)))

    def residual(theta):
        pred = knee_model_log10(f, theta[0], theta[1], theta[2])
        return pred - y

    try:
        res = least_squares(
            residual,
            x0=np.array([offset0, log_knee0, exponent0], dtype=float),
            bounds=([-20.0, np.log(1e-8), 0.05], [20.0, np.log(1e8), 8.0]),
            loss='soft_l1',
            f_scale=0.08,
            max_nfev=300,
        )
        offset, log_knee, exponent = res.x
        knee = float(np.exp(log_knee))
        knee_frequency = float(np.power(knee, 1.0 / exponent)) if knee > 0 and exponent > 0 else np.nan
        timescale_s = float(1.0 / (2.0 * np.pi * knee_frequency)) if knee_frequency > 0 else np.nan
        yhat = knee_model_log10(f, offset, log_knee, exponent)
        ss_res = float(np.nansum((y - yhat) ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.nanmean((y - yhat) ** 2)))
        return float(offset), knee, float(exponent), knee_frequency, timescale_s, r2, rmse, bool(res.success)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False


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
    Y = np.array(f['NormativeData_raw']['Y_model_ready']).T  # subjects x features
    subj = subject_info_model_ready(f, 'NormativeData_raw')
    freq = np.array(f['NormativeData_raw']['freq_grid']).ravel()
    n_sub = Y.shape[0]
    abs_power = np.full((freq.size, N_REGIONS, n_sub), np.nan, dtype=np.float64)
    exponent = np.full((N_REGIONS, n_sub), np.nan, dtype=np.float64)
    offset = np.full((N_REGIONS, n_sub), np.nan, dtype=np.float64)

    for _, row in feat.iterrows():
        idx = int(row.name)
        r = int(row['region']) - 1
        ft = row['feature_type']
        if ft == 'absolute_power':
            hz = int(float(row['frequency_hz']))
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
    freq = np.array(g['freq_grid']).ravel()
    abs_power = np.array(g['abs_power'])  # freq x region x subject
    exponent = np.array(g['aperiodic_exponent'])
    offset = np.array(g['aperiodic_offset'])
    subj = subject_info_model_ready(f, 'NormativeData_combat')
    return freq, abs_power, exponent, offset, subj


def compute_dataset(name, freq, abs_power, exponent_init, offset_init, subj):
    mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
    fit_freq = freq[mask]
    n_freq, n_regions, n_subjects = abs_power.shape
    records = []
    tau = np.full((n_regions, n_subjects), np.nan, dtype=np.float64)
    knee_freq = np.full((n_regions, n_subjects), np.nan, dtype=np.float64)
    knee_param = np.full((n_regions, n_subjects), np.nan, dtype=np.float64)
    fit_exp = np.full((n_regions, n_subjects), np.nan, dtype=np.float64)
    fit_offset = np.full((n_regions, n_subjects), np.nan, dtype=np.float64)
    fit_r2 = np.full((n_regions, n_subjects), np.nan, dtype=np.float64)
    fit_rmse = np.full((n_regions, n_subjects), np.nan, dtype=np.float64)

    t0 = time.time()
    for s in range(n_subjects):
        if s % 50 == 0:
            print(f'{name}: subject {s+1}/{n_subjects}', flush=True)
        sid = subj.loc[s, 'subject_id'] if 'subject_id' in subj.columns and s < len(subj) else f'sub-{s+1:04d}'
        site = subj.loc[s, 'site'] if 'site' in subj.columns and s < len(subj) else ''
        age = subj.loc[s, 'age'] if 'age' in subj.columns and s < len(subj) else np.nan
        sex = subj.loc[s, 'sex'] if 'sex' in subj.columns and s < len(subj) else ''
        for r in range(n_regions):
            y = abs_power[mask, r, s]
            off, kp, exp, fk, ts, r2, rmse, ok = fit_one(fit_freq, y, exponent_init[r, s], offset_init[r, s])
            fit_offset[r, s] = off
            knee_param[r, s] = kp
            fit_exp[r, s] = exp
            knee_freq[r, s] = fk
            tau[r, s] = ts
            fit_r2[r, s] = r2
            fit_rmse[r, s] = rmse
            records.append({
                'dataset_version': name,
                'subject_index': s + 1,
                'subject_id': sid,
                'site': site,
                'age': age,
                'sex': sex,
                'region': r + 1,
                'knee_parameter': kp,
                'knee_frequency_hz': fk,
                'timescale_s': ts,
                'timescale_ms': ts * 1000.0 if np.isfinite(ts) else np.nan,
                'aperiodic_offset_fit': off,
                'aperiodic_exponent_fit': exp,
                'aperiodic_offset_initial': offset_init[r, s],
                'aperiodic_exponent_initial': exponent_init[r, s],
                'fit_r2': r2,
                'fit_rmse_log10': rmse,
                'fit_success': ok,
                'fit_freq_min_hz': FREQ_MIN,
                'fit_freq_max_hz': FREQ_MAX,
            })
    df = pd.DataFrame.from_records(records)
    df.to_csv(OUT / f'timescale_{name}_long.csv', index=False)

    region_summary = df.groupby('region').agg(
        n=('timescale_ms', 'count'),
        timescale_ms_mean=('timescale_ms', 'mean'),
        timescale_ms_sd=('timescale_ms', 'std'),
        timescale_ms_median=('timescale_ms', 'median'),
        knee_frequency_hz_mean=('knee_frequency_hz', 'mean'),
        knee_frequency_hz_median=('knee_frequency_hz', 'median'),
        fit_r2_mean=('fit_r2', 'mean'),
        fit_rmse_log10_mean=('fit_rmse_log10', 'mean'),
    ).reset_index()
    region_summary.to_csv(OUT / f'timescale_{name}_region_summary.csv', index=False)

    subject_summary = df.groupby(['subject_index', 'subject_id', 'site', 'age', 'sex']).agg(
        timescale_ms_mean=('timescale_ms', 'mean'),
        timescale_ms_median=('timescale_ms', 'median'),
        knee_frequency_hz_mean=('knee_frequency_hz', 'mean'),
        fit_r2_mean=('fit_r2', 'mean'),
    ).reset_index()
    subject_summary.to_csv(OUT / f'timescale_{name}_subject_summary.csv', index=False)

    savemat(OUT / f'timescale_{name}.mat', {
        'timescale_s_region_x_subject': tau,
        'timescale_ms_region_x_subject': tau * 1000.0,
        'knee_frequency_hz_region_x_subject': knee_freq,
        'knee_parameter_region_x_subject': knee_param,
        'aperiodic_exponent_fit_region_x_subject': fit_exp,
        'aperiodic_offset_fit_region_x_subject': fit_offset,
        'fit_r2_region_x_subject': fit_r2,
        'fit_rmse_log10_region_x_subject': fit_rmse,
        'fit_freq_range_hz': np.array([FREQ_MIN, FREQ_MAX], dtype=float),
    }, do_compression=True)
    print(f'{name}: wrote outputs in {OUT} ({time.time()-t0:.1f}s)', flush=True)
    return df


def main():
    meta = {
        'source_mat': str(MAT_PATH),
        'method': 'Fit knee aperiodic model log10(P)=offset-log10(knee+f^exponent); knee_frequency=knee^(1/exponent); timescale=1/(2*pi*knee_frequency).',
        'reference': 'Gao et al., Neuronal timescales are functionally dynamic and shaped by cortical microarchitecture, eLife 2020;9:e61277.',
        'fit_frequency_range_hz': [FREQ_MIN, FREQ_MAX],
        'note': 'Input FOOOF/specparam results only included fixed aperiodic exponent/offset, not knee. Therefore knee was refit from PSD using exponent/offset as initialization.',
    }
    (OUT / 'timescale_method_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    with h5py.File(MAT_PATH, 'r') as f:
        freq, abs_power, exp0, off0, subj = raw_region_values(f)
        compute_dataset('RAW_model_ready', freq, abs_power, exp0, off0, subj)
        freq, abs_power, exp0, off0, subj = combat_region_values(f)
        compute_dataset('ComBat_model_ready', freq, abs_power, exp0, off0, subj)

if __name__ == '__main__':
    main()
