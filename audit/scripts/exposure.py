"""Test the paper's claim that 'every high-Sharpe result is long-only exposure in disguise'."""
import numpy as np
import pandas as pd

from cryptoforecast.backtest.costs import turnover
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.evaluate.stats import newey_west_lrv, sharpe_ratio

c = DEFAULT_CONFIG.costs.cost_per_side
df = pd.read_csv("audit/forecasts.csv", parse_dates=["date"])
print(f"{'asset':5}{'h':>3} {'model':14}{'Sharpe':>8}{'B&H':>7}{'%long':>7}{'beta_to_asset':>15}{'alpha_ann':>11}{'t(alpha) NW':>13}")
for (a,h),g0 in df.groupby(["asset","horizon"]):
    piv=g0.pivot_table(index="date",columns="model",values=["y_true","y_pred"])
    y=piv[("y_true","ridge")].to_numpy()
    for m in ("historical_mean","ridge","elastic_net","gbm"):
        d=slice(None)
        yp=piv[("y_pred",m)].to_numpy()[::h]; yy=y[::h]
        pos=np.sign(yp); mkt=np.expm1(yy)
        net=pos*mkt-turnover(pd.Series(pos)).to_numpy()*c
        bh =mkt-turnover(pd.Series(np.ones_like(pos))).to_numpy()*c
        X=np.column_stack([np.ones_like(mkt),bh])
        beta,*_=np.linalg.lstsq(X,net,rcond=None)
        resid=net-X@beta
        se=np.sqrt(newey_west_lrv(resid,2)/resid.size)/np.sqrt(np.mean((bh-bh.mean())**2)*0+1)  # se of intercept approx
        # proper NW se for the intercept
        XtXi=np.linalg.inv(X.T@X); S=np.zeros((2,2))
        for k in range(0,3):
            w=1-k/3
            for t in range(k,len(mkt)):
                u=np.outer(X[t]*resid[t],X[t-k]*resid[t-k]); S+= w*(u+u.T)/2 if k>0 else u
        V=XtXi@S@XtXi; t_alpha=beta[0]/np.sqrt(V[0,0])
        print(f"{a:5}{h:>3} {m:14}{sharpe_ratio(net,365/h):>8.2f}{sharpe_ratio(bh,365/h):>7.2f}"
              f"{100*np.mean(pos>0):>7.0f}{beta[1]:>15.2f}{beta[0]*365/h:>11.3f}{t_alpha:>13.2f}")
