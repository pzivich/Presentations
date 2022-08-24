# Loading data
import pandas as pd
d = pd.read_csv("example_data.csv")
d.info()

# Targeted Maximum Likelihood Estimation
from zepid.causal.doublyrobust import TMLE

tmle = TMLE(d, exposure="a", outcome="y")
tmle.exposure_model("x + z + x:z", bound=0.01)
tmle.outcome_model("a + x + z + a:x")
tmle.fit()
tmle.summary(decimal=2)

# Logistic Regression
import statsmodels.api as sm
import statsmodels.formula.api as smf

fm = smf.glm("y ~ x + z",
             d,
             family=sm.families.Binomial()).fit()
print(fm.summary())

# Inverse Probability Weighting
fm = smf.glm("a ~ x_1 + x_2 + x_3",
             d,
             family=sm.families.Binomial()).fit()
pi_a = fm.predict()

ipw = 1 / (d['a'] * pi_a + (1-d['a'])*(1-pi_a))

f = sm.families.family.Binomial(sm.families.links.identity())
msm = smf.gee("y ~ a", d.index, d,
              weights=ipw,
              family=f).fit()
print(msm.summary())

# Cox Proportional Hazards
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(d[['time', 'delta', 'a', 'z']],
        duration_col='time',
        event_col='delta',
        strata='x')
cph.print_summary()
