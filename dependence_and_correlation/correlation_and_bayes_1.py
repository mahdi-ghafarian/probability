import pandas as pd
import numpy as np

# This script generates two binary variables A and B, calculates their probabilities,
# conditional probabilities, and correlation.

# Parameters
n = 1000 # Number of samples
prob_A = 0.5 # Default probability for A
prob_B = 0.5 # Default probability for B

# Generate random 0/1 values based on probabilities
A = np.random.choice([0, 1], size=n, p=[1 - prob_A, prob_A])
B = np.random.choice([0, 1], size=n, p=[1 - prob_B, prob_B])

# Create DataFrame
df = pd.DataFrame({'A': A, 'B': B})

# Calculate probabilities
P_A = df['A'].mean()
P_B = df['B'].mean()
P_A_and_B = ((df['A'] == 1) & (df['B'] == 1)).mean()
P_A_given_B = P_A_and_B / P_B if P_B > 0 else 0
P_B_given_A = P_A_and_B / P_A if P_A > 0 else 0

# Calculate correlation
correlation = df['A'].corr(df['B'])

# Display results
print("\nResults:")
print(f"P(A) = {P_A:.4f}")
print(f"P(B) = {P_B:.4f}")
print(f"P(A AND B) = {P_A_and_B:.4f}")
print(f"P(A | B) = {P_A_given_B:.4f}")
print(f"P(B | A) = {P_B_given_A:.4f}")
print(f"Correlation between A and B = {correlation:.4f}")
