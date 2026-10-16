from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import render_dk68_brainmaps_last_reference_views as renderer

stats=pd.read_csv(OUT/"MEG_DK68_timescale_age_regionwise_abs_0to500ms_stats.csv").sort_values("region")
beta=stats["age_beta_ms_per_year_adjusted"].to_numpy(float)
sig=stats["age_q_fdr_68"].to_numpy(float)<.05
signed=np.where(sig,beta,np.nan)
positive=np.where(sig&(beta>0),beta,np.nan)
negative=np.where(sig&(beta<0),beta,np.nan)
lim=np.ceil(np.nanmax(np.abs(signed))*10)/10

renderer.OUT=OUT
jobs=[
("MEG_DK68_timescale_age_beta_FDR_signed_brainmap",signed),
("MEG_DK68_timescale_age_beta_FDR_positive_brainmap",positive),
("MEG_DK68_timescale_age_beta_FDR_negative_brainmap",negative)]
for stem,values in jobs:
    renderer.render_plate(values,stem,diverging=True,colorbar=False,fixed_lim=lim)
    renderer.render_plate(values,stem+"_with_colorbar",diverging=True,colorbar=True,
                          fixed_lim=lim,cbar_label="Age beta (ms/year)")
    print("wrote",stem)

source=stats[["region","region_name","n_included_abs_0to500",
              "age_beta_ms_per_year_adjusted","age_p_adjusted_model",
              "age_q_fdr_68","significant_fdr","direction"]].copy()
source["brainmap_signed_beta"]=signed
source["brainmap_positive_beta"]=positive
source["brainmap_negative_beta"]=negative
source.to_csv(OUT/"MEG_DK68_timescale_age_FDR_brainmap_source.csv",index=False)
pd.DataFrame([{"color_min":-lim,"color_center":0.0,"color_max":lim,
               "n_positive":int(np.sum(sig&(beta>0))),
               "n_negative":int(np.sum(sig&(beta<0)))}]).to_csv(
    OUT/"MEG_DK68_timescale_age_FDR_brainmap_color_limits.csv",index=False)

