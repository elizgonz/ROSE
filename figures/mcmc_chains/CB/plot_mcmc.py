import pandas as pd
import numpy as np
import corner
import matplotlib.pyplot as plt

# Load chain
df = pd.read_csv("chain_checkpoint_CB_single.csv")
df_continued = pd.read_csv("chain_checkpoint_CB_single_continued.csv")

# Select parameters
params = ["B_CB","log_h_CB"]

fig, axes = plt.subplots(len(params), 1, figsize=(8, 10), sharex=True)
# x-axis for original chain
x1 = np.arange(len(df))
x2 = np.arange(len(df), len(df) + len(df_continued))
for i, p in enumerate(params):
    axes[i].plot(x1, df[p], alpha=0.6)
    axes[i].plot(x2, df_continued[p], alpha=0.6)
    axes[i].set_ylabel(p)
axes[-1].set_xlabel("Step")
plt.tight_layout()
plt.savefig("trace_plot.png", dpi=200)
plt.clf()

# Remove burn-in (first 20%)
burnin = int(0.2 * len(df))
df_post = df.iloc[burnin:]
# Drop NaNs just in case
df_post = df_post[params].dropna()

# Combine chains
samples = np.vstack((df_post.values, df_continued[params].dropna().values))

# Make corner plot
fig = corner.corner(
    samples,
    labels=params,
    range=[(samples[:,i].min()-1e-6, samples[:,i].max()+1e-6) for i in range(samples.shape[1])],
    show_titles=True,
    title_fmt=".3g",
    quantiles=[0.16, 0.5, 0.84]
    # central 68% credible interval, which behaves like a 1σ uncertainty for roughly Gaussian posteriors
)
plt.savefig("corner_plot.png", dpi=200)
plt.show()