"""(1) Pesaran-Timmermann under overlapping labels. (2) Backtest under feasible execution."""
import numpy as np, pandas as pd
from cryptoforecast.evaluate.stats import pesaran_timmermann, sharpe_ratio, block_bootstrap_ci
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.backtest.costs import turnover

df = pd.read_csv("audit/forecasts.csv", parse_dates=["date"])
ML=("ridge","elastic_net","gbm"); c=DEFAULT_CONFIG.costs.cost_per_side

print("="*108)
print("PESARAN-TIMMERMANN: paper uses all n overlapping h-step forecasts; labels overlap h-1 times")
print("="*108)
print(f"{'asset':5}{'h':>3} {'model':13}{'n_used':>7}{'PT_paper':>10}{'p':>8} | {'n_eff':>6}{'PT_nonovlp(mean over h phases)':>32}{'p_mean':>9}{'worst_phase_p':>15}")
for (a,h,m),g in df[df.model.isin(ML)].groupby(["asset","horizon","model"]):
    g=g.sort_values("date"); y=g.y_true.to_numpy(); p=g.y_pred.to_numpy()
    full=pesaran_timmermann(pd.Series(y),p)
    ph=[pesaran_timmermann(pd.Series(y[k::h]),p[k::h]) for k in range(h)]
    stats_=[x.statistic for x in ph]; ps=[x.p_value for x in ph]
    print(f"{a:5}{h:>3} {m:13}{len(y):>7}{full.statistic:>10.2f}{full.p_value:>8.3f} | {len(y)//h:>6}{np.mean(stats_):>32.2f}{np.mean(ps):>9.3f}{max(ps):>15.3f}")

print()
print("="*108)
print("EXECUTION TIMING: features use the completed bar-t close C_t; the paper's backtest also")
print("enters at C_t and exits at C_{t+h}.  Below: same forecasts, entry delayed to the next close.")
print("="*108)
print(f"{'asset':5}{'h':>3} {'model':13}{'Sharpe_paper(t->t+h)':>22}{'Sharpe_delayed(t+1->t+h+1)':>28}{'delta':>8}")
for (a,h,m),g in df[df.model.isin(ML)].groupby(["asset","horizon","model"]):
    g=g.sort_values("date").reset_index(drop=True)
    def bt(shift):
        d=g.iloc[shift::h] if shift==0 else g.iloc[0:len(g)-shift:h]
        pos=np.sign(d.y_pred.to_numpy())
        if shift==0: real=d.y_true.to_numpy()
        else:
            # realised return over (t+shift, t+h+shift]: rebuild from the close column
            cl=g["close"].to_numpy(); idx=d.index.to_numpy()
            ok=(idx+shift+h)<len(cl); idx=idx[ok]; pos=pos[:len(idx)]
            real=np.log(cl[idx+shift+h]/cl[idx+shift])
        net=pos*np.expm1(real)-turnover(pd.Series(pos)).to_numpy()*c
        return sharpe_ratio(net,365.0/h)
    s0,s1=bt(0),bt(1)
    print(f"{a:5}{h:>3} {m:13}{s0:>22.2f}{s1:>28.2f}{s1-s0:>8.2f}")
