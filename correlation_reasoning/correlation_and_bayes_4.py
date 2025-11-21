import pandas as pd
import numpy as np
import plotly.express as px

#------------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------------
n = 1000          # number of samples
p_A = 0.2         # probability of event A being true (A = 1), e.g. cloudy today
p_B = 0.3          # probability of event B being true (B = 1), e.g. raining today
rho = 0.1        # desired correlation between A and B

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
# # Bar Chart of Joint Outcome Frequencies
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
# Joint Probability Chart
# as rectangles with areas proportional to joint probabilities
#--------------------------------------------------------------------------------------

# Interpretation of conditional probabilities:
# (A|B) =(BA)/(B)=(BA)/(BA + B~A) = p11/(P11 + P01)
# (B|A) =(AB)/(A)=(AB)/(AB + A~B) = p11/(P11 + P10)


from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Existing calculations remain unchanged ---
freqs = {
    'P00': int(P00 * n),
    'P01': int(P01 * n),
    'P10': int(P10 * n),
    'P11': int(P11 * n),
}

P_B0 = P00 + P10
P_B1 = P01 + P11
P_A0 = P00 + P01
P_A1 = P10 + P11

P_A0_given_B0 = P00 / P_B0
P_A1_given_B0 = P10 / P_B0
P_A0_given_B1 = P01 / P_B1
P_A1_given_B1 = P11 / P_B1

P_B0_given_A0 = P00 / P_A0
P_B1_given_A0 = P01 / P_A0
P_B0_given_A1 = P10 / P_A1
P_B1_given_A1 = P11 / P_A1

# ✅ Color scheme for joint regions
colors = {
    'P00': '#A8E6CF',  # mint green
    'P01': '#FFD3B6',  # peach orange
    'P10': '#D1C4E9',  # lavender blue
    'P11': '#FFB6C1',  # rose pink
}

# Combine all 8 conditional probabilities into one list
labels_all = [
    '~A|~B', 'A|~B', '~A|B', 'A|B',  # P(A|B)
    '~B|~A', 'B|~A', '~B|A', 'B|A'   # P(B|A)
]
values_all = [
    P_A0_given_B0, P_A1_given_B0, P_A0_given_B1, P_A1_given_B1,
    P_B0_given_A0, P_B1_given_A0, P_B0_given_A1, P_B1_given_A1
]

sorted_all = sorted(zip(values_all, labels_all), reverse=True)

# Ratios chart data
ratios_labels = ['P(A|B)/P(A)', 'P(A|~B)/P(A)', 'P(~B|A)/P(~B)', 'P(B|A)/P(B)',
                 'P(~A|B)/P(~A)', 'P(~A|~B)/P(~A)', 'P(B|~A)/P(B)', 'P(~B|~A)/P(~B)']
ratios_values = [
    P_A1_given_B1 / p_A,
    P_A1_given_B0 / p_A,
    P_B0_given_A1 / (1 - p_B),
    P_B1_given_A1 / p_B,
    (1 - P_A1_given_B1) / (1 - p_A),
    (1 - P_A1_given_B0) / (1 - p_A),
    P_B1_given_A0 / p_B,
    P_B0_given_A0 / (1 - p_B)
]

# Sort ratios descending
sorted_ratios = sorted(zip(ratios_values, ratios_labels), reverse=True)

# ✅ Map labels to joint regions for first chart
joint_map_first = {
    '~A|~B': 'P00',
    'A|~B': 'P10',
    '~A|B': 'P01',
    'A|B': 'P11',
    '~B|~A': 'P00',
    'B|~A': 'P01',
    '~B|A': 'P10',
    'B|A': 'P11'
}

first_chart_colors = [colors[joint_map_first[label]] for _, label in sorted_all]

# ✅ Colors for ratios chart based on numerator logic
color_map = {
    'P(A|B)/P(A)': colors['P11'],
    'P(A|~B)/P(A)': colors['P10'],
    'P(~B|A)/P(~B)': colors['P10'],
    'P(B|A)/P(B)': colors['P11'],
    'P(~A|B)/P(~A)': colors['P01'],
    'P(~A|~B)/P(~A)': colors['P00'],
    'P(B|~A)/P(B)': colors['P01'],
    'P(~B|~A)/P(~B)': colors['P00']
}
ratio_colors = [color_map[label] for _, label in sorted_ratios]

# -------------------------------
# ✅ FIGURE 1: Bar Charts
# -------------------------------
fig1 = make_subplots(rows=1, cols=2,
                     subplot_titles=("Posterior/Prior Ratios", "Conditional Probability"),
                     column_widths=[0.5, 0.5])

# Panel 1: Ratio Chart
fig1.add_trace(go.Bar(
    x=[label for _, label in sorted_ratios],
    y=[val for val, _ in sorted_ratios],
    marker_color=ratio_colors,
    text=[f"{val:.2f}" for val, _ in sorted_ratios],
    textposition='auto',
    showlegend=False
), row=1, col=1)

# Horizontal line for ratio chart (y=1)
fig1.add_shape(type="line", x0=-0.5, x1=7.5, y0=1, y1=1,
               line=dict(color="grey", dash="dot", width=1), xref='x1', yref='y1')

# Panel 2: Conditional Probability Chart
fig1.add_trace(go.Bar(
    x=[label for _, label in sorted_all],
    y=[val for val, _ in sorted_all],
    marker_color=first_chart_colors,
    text=[f"{val:.2f}" for val, _ in sorted_all],
    textposition='auto',
    showlegend=False
), row=1, col=2)

# Horizontal reference lines for conditional probability chart
fig1.add_shape(type="line", x0=-0.5, x1=7.5, y0=p_A, y1=p_A,
               line=dict(color="grey", dash="dot", width=1), xref='x2', yref='y2')
fig1.add_shape(type="line", x0=-0.5, x1=7.5, y0=1-p_A, y1=1-p_A,
               line=dict(color="grey", dash="dot", width=1), xref='x2', yref='y2')
fig1.add_shape(type="line", x0=-0.5, x1=7.5, y0=p_B, y1=p_B,
               line=dict(color="grey", dash="dot", width=1), xref='x2', yref='y2')
fig1.add_shape(type="line", x0=-0.5, x1=7.5, y0=1-p_B, y1=1-p_B,
               line=dict(color="grey", dash="dot", width=1), xref='x2', yref='y2')

fig1.update_layout(
    title=f"Conditional Probabilities - P(A)={p_A:.2f}, P(B)={p_B:.2f}, ρ={rho:.2f}",
    width=1600,
    height=600,
    plot_bgcolor='white',
    paper_bgcolor='white'
)
fig1.show()


# -------------------------------
# ✅ FIGURE 2: Area Charts (EXACTLY AS BEFORE)
# -------------------------------
fig2 = make_subplots(rows=1, cols=2,
                     subplot_titles=("P(A|B)", "P(B|A)"),
                     column_widths=[0.5, 0.5])

# Panel 1: P(A|B) rectangles
x0 = 0
x1 = P_B0
x2 = x1 + P_B1
fig2.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=P_A0_given_B0, fillcolor=colors['P00'], line=dict(color="black"), row=1, col=1)
fig2.add_shape(type="rect", x0=x0, x1=x1, y0=P_A0_given_B0, y1=1, fillcolor=colors['P10'], line=dict(color="black"), row=1, col=1)
fig2.add_shape(type="rect", x0=x1, x1=x2, y0=0, y1=P_A0_given_B1, fillcolor=colors['P01'], line=dict(color="black"), row=1, col=1)
fig2.add_shape(type="rect", x0=x1, x1=x2, y0=P_A0_given_B1, y1=1, fillcolor=colors['P11'], line=dict(color="black"), row=1, col=1)

# Annotations for P(A|B)
fig2.add_annotation(x=(x0 + x1)/2, y=P_A0_given_B0/2,
    text=f"P00 = {P00:.2f}<br>P(~A|~B) = {P_A0_given_B0:.2f}<br>N = {freqs['P00']}",
    showarrow=False, font=dict(color="black", size=12), row=1, col=1)
fig2.add_annotation(x=(x0 + x1)/2, y=(P_A0_given_B0 + 1)/2,
    text=f"P10 = {P10:.2f}<br>P(A|~B) = {P_A1_given_B0:.2f}<br>N = {freqs['P10']}",
    showarrow=False, font=dict(color="black", size=12), row=1, col=1)
fig2.add_annotation(x=(x1 + x2)/2, y=P_A0_given_B1/2,
    text=f"P01 = {P01:.2f}<br>P(~A|B) = {P_A0_given_B1:.2f}<br>N = {freqs['P01']}",
    showarrow=False, font=dict(color="black", size=12), row=1, col=1)
fig2.add_annotation(x=(x1 + x2)/2, y=(P_A0_given_B1 + 1)/2,
    text=f"P11 = {P11:.2f}<br>P(A|B) = {P_A1_given_B1:.2f}<br>N = {freqs['P11']}",
    showarrow=False, font=dict(color="black", size=12), row=1, col=1)

# Panel 2: P(B|A) rectangles
y0 = 0
y1 = P_A0
y2 = y1 + P_A1
fig2.add_shape(type="rect", x0=0, x1=P_B0_given_A0, y0=y0, y1=y1, fillcolor=colors['P00'], line=dict(color="black"), row=1, col=2)
fig2.add_shape(type="rect", x0=P_B0_given_A0, x1=1, y0=y0, y1=y1, fillcolor=colors['P01'], line=dict(color="black"), row=1, col=2)
fig2.add_shape(type="rect", x0=0, x1=P_B0_given_A1, y0=y1, y1=y2, fillcolor=colors['P10'], line=dict(color="black"), row=1, col=2)
fig2.add_shape(type="rect", x0=P_B0_given_A1, x1=1, y0=y1, y1=y2, fillcolor=colors['P11'], line=dict(color="black"), row=1, col=2)

# Annotations for P(B|A)
fig2.add_annotation(x=P_B0_given_A0/2, y=(y0 + y1)/2,
    text=f"P00 = {P00:.2f}<br>P(~B|~A) = {P_B0_given_A0:.2f}<br>N = {freqs['P00']}",
    showarrow=False, font=dict(color="black", size=12), row=1, col=2)
fig2.add_annotation(x=(P_B0_given_A0 + 1)/2, y=(y0 + y1)/2,
    text=f"P01 = {P01:.2f}<br>P(B|~A) = {P_B1_given_A0:.2f}<br>N = {freqs['P01']}",
    showarrow=False, font=dict(color="black", size=12), row=1, col=2)
fig2.add_annotation(x=P_B0_given_A1/2, y=(y1 + y2)/2,
    text=f"P10 = {P10:.2f}<br>P(~B|A) = {P_B0_given_A1:.2f}<br>N = {freqs['P10']}",
    showarrow=False, font=dict(color="black", size=12), row=1, col=2)
fig2.add_annotation(x=(P_B0_given_A1 + 1)/2, y=(y1 + y2)/2,
    text=f"P11 = {P11:.2f}<br>P(B|A) = {P_B1_given_A1:.2f}<br>N = {freqs['P11']}",
    showarrow=False, font=dict(color="black", size=12), row=1, col=2)

# Hide tick labels for rectangle charts
fig2.update_xaxes(showticklabels=False, row=1, col=1)
fig2.update_yaxes(showticklabels=False, row=1, col=1)
fig2.update_xaxes(showticklabels=False, row=1, col=2)
fig2.update_yaxes(showticklabels=False, row=1, col=2)

fig2.update_layout(
    title=f"Conditional Probabilities - P(A)={p_A:.2f}, P(B)={p_B:.2f}, ρ={rho:.2f}",
    width=1600,
    height=600,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Fix axis ranges so shapes fill the space
fig2.update_xaxes(range=[0, P_B0 + P_B1], row=1, col=1)
fig2.update_yaxes(range=[0, 1], row=1, col=1)
fig2.update_xaxes(range=[0, 1], row=1, col=2)
fig2.update_yaxes(range=[0, P_A0 + P_A1], row=1, col=2)

fig2.show()
