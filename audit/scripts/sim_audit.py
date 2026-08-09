"""Audit the paper's Clark-West 'validation' simulation (tests/test_stats.py:103)."""
import numpy as np
import pandas as pd

from cryptoforecast.evaluate.stats import clark_west, diebold_mariano

rng = np.random.default_rng(4)
sy, su, n, R = 0.03, 0.01, 250, 5000
print("PAPER'S SIMULATION:  y ~ N(0, %.2f),  model forecast = u ~ N(0, %.2f) independent of y,  benchmark = 0" % (sy,su))
print()
print("Population MSPE, analytically:")
print("  benchmark : E[(y-0)^2]  = sy^2         = %.6f" % sy**2)
print("  model     : E[(y-u)^2]  = sy^2 + su^2  = %.6f" % (sy**2+su**2))
print("  difference (model worse by)            = %.6f   <-- the null of equal accuracy is FALSE" % su**2)
print()
mspe_b, mspe_m, dms, cws, dbar = [], [], [], [], []
for _ in range(R):
    y = rng.normal(0.0, sy, n); u = rng.normal(0.0, su, n); b = np.zeros(n)
    ys = pd.Series(y)
    mspe_b.append(np.mean((y-b)**2)); mspe_m.append(np.mean((y-u)**2))
    dbar.append(np.mean((y-u)**2 - (y-b)**2))
    dms.append(diebold_mariano(ys,u,b).statistic); cws.append(clark_west(ys,u,b).statistic)
dms, cws = np.array(dms), np.array(cws)
print("Monte Carlo over %d replications (n=%d):" % (R,n))
print("  mean sample MSPE benchmark = %.6f" % np.mean(mspe_b))
print("  mean sample MSPE model     = %.6f" % np.mean(mspe_m))
print("  mean loss differential     = %.6f  (analytic su^2 = %.6f)" % (np.mean(dbar), su**2))
print("  mean DM statistic          = %+.3f   -> DM is detecting a model that IS genuinely worse")
print("  DM rejects 'model worse' at 5%%: %.1f%%  <- POWER against a false null, not size distortion" % (100*np.mean(dms>1.96)))
print("  mean CW statistic          = %+.3f" % cws.mean())
print("  CW rejects at one-sided 5%%: %.1f%%" % (100*np.mean(cws>1.645)))
print()
print("No parameter is estimated anywhere in this simulation: 'u' is exogenous noise, not a fitted")
print("forecast. The estimation-noise term that Clark-West exists to remove is therefore absent,")
print("and the experiment cannot measure the size of either test under a nested-estimation null.")
print()
print("Algebraic identity behind the apparent 'centering' of CW:")
print("  f_t = (y-b)^2 - [(y-m)^2 - (b-m)^2] = 2(y_t - b_t)(m_t - b_t);  with b=0 this is 2*y_t*u_t,")
print("  and E[2 y u] = 0 for ANY independent u, however bad a forecast u is.")
f = 2*rng.normal(0,sy,200000)*rng.normal(0,su,200000)
print("  numerical check E[2*y*u] = %+.3e (se %.1e)" % (f.mean(), f.std()/np.sqrt(f.size)))
