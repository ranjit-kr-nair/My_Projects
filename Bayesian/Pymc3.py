import pandas as pd
import pymc3 as pm

data = pd.read_csv('data.csv')
test_data = data[data['group'] == 'test']['conversion'].values
control_data = data[data['group'] == 'control']['conversion'].values

with pm.Model() as model:
    # Priors
    p_control = pm.Uniform('p_control', 0, 1)
    p_test = pm.Uniform('p_test', 0, 1)

    # Likelihoods
    control = pm.Bernoulli('control', p=p_control, observed=control_data)
    test = pm.Bernoulli('test', p=p_test, observed=test_data)

    # Difference
    diff = pm.Deterministic('diff', p_test - p_control)

    # Inference
    trace = pm.sample(draws=2000, tune=1000)

prob_test_beats_control = (trace['diff'] > 0).mean()
print('Probability of test group beating control group:', prob_test_beats_control)
