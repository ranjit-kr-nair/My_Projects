import pandas as pd
import scipy.stats as stats

# Import data from CSV file
data = pd.read_csv('data.csv')

# Split data into test and control groups
test = data[data['group'] == 'test']
control = data[data['group'] == 'control']

# Calculate conversion rates for test and control groups
test_conv_rate = test['did the user convert'].mean()
control_conv_rate = control['did the user convert'].mean()

# Calculate the probability of the test group beating the control group
p_value = stats.ttest_ind(test['did the user convert'], control['did the user convert']).pvalue / 2  # Divide by 2 for one-tailed test
test_beats_control_prob = 1 - p_value

print('Test conversion rate:', test_conv_rate)
print('Control conversion rate:', control_conv_rate)
print('Probability of test beating control:', test_beats_control_prob)
