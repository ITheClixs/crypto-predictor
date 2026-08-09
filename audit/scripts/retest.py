"""Re-run the paper's inference with (a) the correct nested benchmark and (b) honest HAC bandwidths."""
import math

import numpy as np
import pandas as pd
from scipy import stats

from cryptoforecast.evaluate.stats import benjamini_hochberg_adjusted, holm_adjusted, newey_west_lrv

df = pd.read_csv("audit/forecasts.csv", parse_dates=["date"])
ML = ("ridge","elastic_net","gbm")

def cw(y, m, b, lags):
    f = 2.0*(y-b)*(m-b)                     # algebraically identical to the CW form
    lrv = newey_west_lrv(f, lags)
    stat = f.mean()/math.sqrt(lrv/f.size)
    return stat, 1.0-stats.norm.cdf(stat), f

rows=[]
for (a,h), g in df.groupby(["asset","horizon"]):
    piv = g.pivot_table(index="date", columns="model", values=["y_true","y_pred"])
    y   = piv[("y_true","ridge")].to_numpy()
    zero= np.zeros_like(y)
    mean_b = piv[("y_pred","historical_mean")].to_numpy()
    n = y.size
    lag_paper = max(0,h-1)
    lag_nw    = int(np.floor(4*(n/100)**(2/9)))      # Newey-West plug-in rule
    lag_cube  = int(round(n**(1/3)))
    for mdl in ML:
        m = piv[("y_pred",mdl)].to_numpy()
        s0,p0,f0 = cw(y,m,zero,lag_paper)            # paper's test
        s0n,p0n,_= cw(y,m,zero,lag_nw)
        s1,p1,_  = cw(y,m,mean_b,lag_paper)          # correct nested benchmark
        s1n,p1n,_= cw(y,m,mean_b,lag_nw)
        s1c,p1c,_= cw(y,m,mean_b,lag_cube)
        # decomposition of the paper's statistic: E[y*m] = E[y]E[m] + Cov(y,m)
        drift = y.mean()*m.mean(); cov = np.cov(y,m,ddof=1)[0,1]
        rows.append(dict(asset=a,h=h,model=mdl,n=n,
            cw_zero=s0,p_zero=p0, cw_zero_nwlag=s0n,p_zero_nw=p0n,
            cw_mean=s1,p_mean=p1, cw_mean_nwlag=s1n,p_mean_nw=p1n, cw_mean_cube=s1c,p_mean_cube=p1c,
            drift_share=drift/(drift+cov), lag_paper=lag_paper, lag_nw=lag_nw, lag_cube=lag_cube))
    # benchmarks' own CW vs zero: features contribute nothing here by construction
    for mdl in ("historical_mean","ar1"):
        m = piv[("y_pred",mdl)].to_numpy()
        s0,p0,_ = cw(y,m,zero,lag_paper)
        rows.append(dict(asset=a,h=h,model=mdl+" (BENCHMARK)",n=n,cw_zero=s0,p_zero=p0,
                         drift_share=(y.mean()*m.mean())/(y.mean()*m.mean()+np.cov(y,m,ddof=1)[0,1])))

r = pd.DataFrame(rows)
ml = r[r.model.isin(ML)].copy()
for col_in,col_out in [("p_zero","holm_zero"),("p_mean","holm_mean"),("p_mean_nw","holm_mean_nw")]:
    ml[col_out]=holm_adjusted(ml[col_in].to_numpy())
    ml[col_out.replace("holm","bh")]=benjamini_hochberg_adjusted(ml[col_in].to_numpy())

pd.set_option("display.width",250,"display.max_columns",50)
print("="*110); print("CW: paper benchmark (zero) vs correct nested benchmark (recursive mean)"); print("="*110)
print(ml[["asset","h","model","cw_zero","p_zero","cw_mean","p_mean","cw_mean_nwlag","p_mean_nw","drift_share"]].round(4).to_string(index=False))
print()
print("rejections at raw 5%%:  zero-benchmark = %d/18 | mean-benchmark = %d/18 | mean-benchmark + NW lag = %d/18"
      % ((ml.p_zero<.05).sum(), (ml.p_mean<.05).sum(), (ml.p_mean_nw<.05).sum()))
print("survive BH(5%%):        zero = %d | mean = %d | mean+NW = %d"
      % ((ml.bh_zero<.05).sum(), (ml.bh_mean<.05).sum(), (ml.bh_mean_nw<.05).sum()))
print("survive Holm(5%%):      zero = %d | mean = %d | mean+NW = %d"
      % ((ml.holm_zero<.05).sum(), (ml.holm_mean<.05).sum(), (ml.holm_mean_nw<.05).sum()))
print()
print("="*110); print("Benchmark forecasters tested against the paper's own zero benchmark (zero feature information)"); print("="*110)
print(r[~r.model.isin(ML)][["asset","h","model","cw_zero","p_zero","drift_share"]].round(4).to_string(index=False))
ml.to_csv("audit/retest_cw.csv",index=False)
