import pandas as pd
import numpy as np
import plotly.express as px

#------------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------------
n = 10000          # number of samples
p_A = 0.7          # probability of event A being true (A = 1), e.g. cloudy today
p_B = 0.3          # probability of event B being true (B = 1), e.g. raining today
rho = 0.25         # desired correlation between A and B

#------------------------------------------------------------------------------------
# Joint Probability Calculation
#------------------------------------------------------------------------------------

# Compute feasible correlation bounds
# These bounds ensure that joint probabilities remain valid (non-negative)
rho_min = (max(p_A + p_B - 1, 0) - p_A * p_B) / np.sqrt(p_A * (1 - p_A) * p_B * (1 - p_B))
rho_max = (min(p_A, p_B) - p_A * p_B) / np.sqrt(p_A * (1 - p_A) * p_B * (1 - p_B))

# Clip rho to feasible range
rho = np.clip(rho, rho_min, rho_max)

# Compute P11 using correlation formula
# Bivariate Bernoulli Distribution
# See https://www.mystatsmind.com/post/simulating-correlated-bernoulli-data
# P(A=1, B=1) = rho * sqrt(P(A=1) * (1 - P(A=1)) * P(B=1) * (1 - P(B=1)))
cov_term = rho * np.sqrt(p_A * (1 - p_A) * p_B * (1 - p_B))
P11 = cov_term + p_A * p_B

# Compute other joint probabilities
# P(A=1, B=0) = P(A=1) - P(A=1, B=1)
# P(A=0, B=1) = P(B=1) - P(A=1, B=1)
# P(A=0, B=0) = 1 - P(A=1, B=1) - P(A=1, B=0) - P(A=0, B=1)
P10 = p_A - P11
P01 = p_B - P11
P00 = 1 - (P11 + P10 + P01)

# Validate probabilities sum to 1
if abs((P11 + P10 + P01 + P00) - 1) > 1e-8:
    raise ValueError("Joint probabilities do not sum to 1.")

# Validate non-negative probabilities
if any(p < 0 for p in [P11, P10, P01, P00]):
    raise ValueError("Negative probability detected after adjustment.")

#------------------------------------------------------------------------------------
# Sampling from the joint distribution
#------------------------------------------------------------------------------------

# Create contingency table for sampling
outcomes = [(1, 1), (1, 0), (0, 1), (0, 0)]
probs = [P11, P10, P01, P00]

# Sample data
np.random.seed(42) # for reproducibility
# Sample n outcomes based on the joint probabilities
# sampled = [0,0,1,2,3,0,1,2,...]
# the frequency of each outcome corresponds to the joint probabilities (probs)
sampled = np.random.choice(len(outcomes), size=n, p=probs)
print("Sampled outcomes (first 10):", sampled[:10])

# Extract sampled outcomes
# example: sampled[5] = 1
# outcomes[3][0] = 1, outcomes[3][1] = 0
# this means A=1, B=0 for the 5th sampled outcome
A = [outcomes[i][0] for i in sampled] # select first element of each outcome
B = [outcomes[i][1] for i in sampled] # select second element of each outcome

# Create DataFrame
df = pd.DataFrame({'A': A, 'B': B})

# Calculate probabilities
P_A_emp = df['A'].mean()
P_B_emp = df['B'].mean()
P_A_and_B_emp = ((df['A'] == 1) & (df['B'] == 1)).mean()
P_A_given_B_emp = P_A_and_B_emp / P_B_emp if P_B_emp > 0 else 0
P_B_given_A_emp = P_A_and_B_emp / P_A_emp if P_A_emp > 0 else 0
correlation_emp = df['A'].corr(df['B'])

#------------------------------------------------------------------------------------
# Printing probabilities and correlation
#------------------------------------------------------------------------------------
print("\nResults:")
print(f"P(A) = {P_A_emp:.4f}")
print(f"P(B) = {P_B_emp:.4f}")
print(f"P(A AND B) = {P_A_and_B_emp:.4f}")
print(f"P(A | B) = {P_A_given_B_emp:.4f}")
print(f"P(B | A) = {P_B_given_A_emp:.4f}")
print(f"Correlation = {correlation_emp:.4f}")
print(f"Joint Probabilities: P11={P11:.4f}, P10={P10:.4f}, P01={P01:.4f}, P00={P00:.4f}")

#------------------------------------------------------------------------------------
# Summary Statistics for A and B
#------------------------------------------------------------------------------------
print("\n Summary Statistics:")
print(f"A: mean={df['A'].mean():.4f}, var={df['A'].var():.4f}, std={df['A'].std():.4f}")
print(f"B: mean={df['B'].mean():.4f}, var={df['B'].var():.4f}, std={df['B'].std():.4f}")
covariance = np.cov(df['A'], df['B'])[0, 1]
print(f"Covariance(A, B) = {covariance:.4f}")

#------------------------------------------------------------------------------------
# Visualization
#------------------------------------------------------------------------------------

# import plotly.express as px

# # Bar Chart of Joint Outcome Frequencies
# freqs = df.value_counts().reset_index()
# freqs.columns = ['A', 'B', 'count']
# freqs['Outcome'] = freqs.apply(lambda row: f"A={row['A']},B={row['B']}", axis=1)

# fig_bar = px.bar(
#     freqs,
#     x='Outcome',
#     y='count',
#     title="Joint Outcome Frequencies",
#     labels={'Outcome': 'Outcome', 'count': 'Count'}
# )
# fig_bar.show()


#--------------------------------------------------------------------------------------
import plotly.graph_objects as go

# Compute column widths based on joint probabilities
width_A0 = P00 + P01
width_A1 = P10 + P11

# Compute heights within each column
height_A0_B0 = P00 / width_A0
height_A0_B1 = P01 / width_A0
height_A1_B0 = P10 / width_A1
height_A1_B1 = P11 / width_A1

# Compute frequencies from sampled data
freq_11 = ((df['A'] == 1) & (df['B'] == 1)).sum()
freq_10 = ((df['A'] == 1) & (df['B'] == 0)).sum()
freq_01 = ((df['A'] == 0) & (df['B'] == 1)).sum()
freq_00 = ((df['A'] == 0) & (df['B'] == 0)).sum()

fig = go.Figure()

# (A=0,B=0) bottom-left
fig.add_shape(type="rect", x0=0, x1=width_A0, y0=0, y1=height_A0_B0,
              fillcolor="lightgreen", opacity=0.6, line=dict(color="black"))
# (A=0,B=1) top-left
fig.add_shape(type="rect", x0=0, x1=width_A0, y0=height_A0_B0, y1=1,
              fillcolor="green", opacity=0.6, line=dict(color="black"))
# (A=1,B=0) bottom-right
fig.add_shape(type="rect", x0=width_A0, x1=1, y0=0, y1=height_A1_B0,
              fillcolor="lightblue", opacity=0.6, line=dict(color="black"))
# (A=1,B=1) top-right
fig.add_shape(type="rect", x0=width_A0, x1=1, y0=height_A1_B0, y1=1,
              fillcolor="blue", opacity=0.6, line=dict(color="black"))

# ✅ Add labels in requested format with probabilities and frequencies
fig.add_annotation(x=width_A0/2, y=height_A0_B0/2,
                   text=f"(A=0,B=0)<br>P={P00:.3f}<br>N={freq_00}", showarrow=False,
                   font=dict(size=14, color="black"))
fig.add_annotation(x=width_A0/2, y=(height_A0_B0 + 1)/2,
                   text=f"(A=0,B=1)<br>P={P01:.3f}<br>N={freq_01}", showarrow=False,
                   font=dict(size=14, color="white"))
fig.add_annotation(x=(width_A0 + 1)/2, y=height_A1_B0/2,
                   text=f"(A=1,B=0)<br>P={P10:.3f}<br>N={freq_10}", showarrow=False,
                   font=dict(size=14, color="black"))
fig.add_annotation(x=(width_A0 + 1)/2, y=(height_A1_B0 + 1)/2,
                   text=f"(A=1,B=1)<br>P={P11:.3f}<br>N={freq_11}", showarrow=False,
                   font=dict(size=14, color="white"))

# Layout: remove axis ticks
fig.update_xaxes(range=[0, 1], showticklabels=False)
fig.update_yaxes(range=[0, 1], showticklabels=False)
fig.update_layout(title="Joint Probability Chart",
                  width=800, height=600)

fig.show()