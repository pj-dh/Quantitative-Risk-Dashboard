"""
Quantitative Strategy Evaluation and Comparison
================================================
A beginner data science project by a student learning quant finance.

This script simulates stock price data and compares 3 simple trading strategies:
1. Buy and Hold
2. Momentum Strategy
3. Simple Moving Average (SMA) Crossover
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# Make sure the plots folder exists
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Set a nice style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set random seed so results are reproducible
np.random.seed(42)

print("=" * 55)
print("  Quantitative Strategy Evaluation and Comparison")
print("=" * 55)

# -------------------------------------------------------
# STEP 1: Simulate Stock Price Data
# -------------------------------------------------------
# We simulate 5 years of daily prices for 3 stocks
# using a simple random walk (Geometric Brownian Motion)
print("\n[1/6] Simulating stock price data...")

n_days = 1260  # about 5 years of trading days
dates = pd.date_range(start="2019-01-01", periods=n_days, freq="B")

def simulate_stock(start_price, annual_return, annual_vol, n_days, seed=0):
    """Simulate daily stock prices using a random walk."""
    np.random.seed(seed)
    daily_return = annual_return / 252          # convert annual to daily
    daily_vol    = annual_vol    / np.sqrt(252) # convert annual to daily
    
    # Each day's return is a small drift + random shock
    returns = np.random.normal(daily_return, daily_vol, n_days)
    
    # Build price series from cumulative returns
    prices = start_price * np.cumprod(1 + returns)
    return prices

# Simulate three different stocks
price_A = simulate_stock(100, 0.12, 0.20, n_days, seed=1)  # growth stock
price_B = simulate_stock(50,  0.08, 0.15, n_days, seed=2)  # stable stock
price_C = simulate_stock(75,  0.15, 0.30, n_days, seed=3)  # volatile stock

# Put everything in a DataFrame (like a spreadsheet)
prices = pd.DataFrame({
    "Stock_A": price_A,
    "Stock_B": price_B,
    "Stock_C": price_C
}, index=dates)

# Save the data so we can look at it later
prices.to_csv("data/simulated_prices.csv")
print(f"   Saved {n_days} days of price data for 3 stocks.")
print(f"   Date range: {dates[0].date()} to {dates[-1].date()}")

# -------------------------------------------------------
# STEP 2: Calculate Daily Returns
# -------------------------------------------------------
print("\n[2/6] Calculating daily returns...")

# Daily return = (today's price - yesterday's price) / yesterday's price
daily_returns = prices.pct_change().dropna()

print(f"   Average daily return - Stock A: {daily_returns['Stock_A'].mean():.4f}")
print(f"   Average daily return - Stock B: {daily_returns['Stock_B'].mean():.4f}")
print(f"   Average daily return - Stock C: {daily_returns['Stock_C'].mean():.4f}")

# -------------------------------------------------------
# STEP 3: Build the 3 Strategies (using Stock_A only)
# -------------------------------------------------------
print("\n[3/6] Building trading strategies...")

stock = prices["Stock_A"].copy()
ret   = daily_returns["Stock_A"].copy()

# --- Strategy 1: Buy and Hold ---
# Super simple: buy on day 1, hold forever, never sell.
bh_returns = ret.copy()

# --- Strategy 2: Momentum Strategy ---
# Idea: if the stock went up last month, it will keep going up.
# We look back 20 days. If return > 0, we go long (buy). Otherwise, we sit out.
lookback = 20
momentum_signal = ret.rolling(lookback).mean()   # average return over past 20 days
momentum_returns = ret * (momentum_signal > 0).shift(1)  # trade next day based on yesterday's signal
momentum_returns = momentum_returns.dropna()

# --- Strategy 3: Simple Moving Average (SMA) Crossover ---
# Idea: use two moving averages - a fast one (10 days) and a slow one (50 days).
# When the fast MA crosses above the slow MA → buy signal.
# When it crosses below → sell (move to cash, 0% return).
fast_ma = stock.rolling(10).mean()   # short-term average
slow_ma = stock.rolling(50).mean()   # long-term average

# Signal: 1 = buy, 0 = stay in cash
sma_signal = (fast_ma > slow_ma).astype(int)
sma_returns = ret * sma_signal.shift(1)   # use yesterday's signal to trade today
sma_returns = sma_returns.dropna()

# Align all strategies to the same dates
common_index = bh_returns.index.intersection(momentum_returns.index).intersection(sma_returns.index)
bh_returns       = bh_returns.loc[common_index]
momentum_returns = momentum_returns.loc[common_index]
sma_returns      = sma_returns.loc[common_index]

# Package into one DataFrame
all_returns = pd.DataFrame({
    "Buy & Hold":  bh_returns,
    "Momentum":    momentum_returns,
    "SMA Crossover": sma_returns
})

print(f"   Strategies built with {len(all_returns)} trading days.")

# -------------------------------------------------------
# STEP 4: Calculate Performance Metrics
# -------------------------------------------------------
print("\n[4/6] Calculating performance metrics...")

def sharpe_ratio(returns, risk_free=0.02):
    """
    Sharpe Ratio = (Average Return - Risk Free Rate) / Std Deviation
    A higher Sharpe means better risk-adjusted returns.
    Rule of thumb: > 1 is good, > 2 is great.
    """
    daily_rf = risk_free / 252   # convert annual risk-free rate to daily
    excess   = returns - daily_rf
    if excess.std() == 0:
        return 0
    return (excess.mean() / excess.std()) * np.sqrt(252)  # annualize

def max_drawdown(returns):
    """
    Max Drawdown = the biggest peak-to-valley loss.
    For example, -0.30 means the strategy lost 30% from its peak at some point.
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def total_return(returns):
    """Total return over the whole period."""
    return (1 + returns).prod() - 1

def annualized_return(returns):
    """Annualized return (assuming 252 trading days per year)."""
    n_years = len(returns) / 252
    total   = (1 + returns).prod()
    return total ** (1 / n_years) - 1

def volatility(returns):
    """Annualized volatility (standard deviation of daily returns * sqrt(252))."""
    return returns.std() * np.sqrt(252)

# Build a summary table
metrics = {}
for col in all_returns.columns:
    r = all_returns[col]
    metrics[col] = {
        "Total Return":        f"{total_return(r):.1%}",
        "Annual Return":       f"{annualized_return(r):.1%}",
        "Annual Volatility":   f"{volatility(r):.1%}",
        "Sharpe Ratio":        f"{sharpe_ratio(r):.2f}",
        "Max Drawdown":        f"{max_drawdown(r):.1%}",
    }

metrics_df = pd.DataFrame(metrics).T
print("\n   Performance Summary:")
print(metrics_df.to_string())
metrics_df.to_csv("data/performance_metrics.csv")

# -------------------------------------------------------
# STEP 5: Generate Plots
# -------------------------------------------------------
print("\n[5/6] Generating plots...")

# ----- Plot 1: Simulated Stock Prices -----
fig, ax = plt.subplots(figsize=(12, 5))
for col in prices.columns:
    ax.plot(prices.index, prices[col], label=col, linewidth=1.5)

ax.set_title("Simulated Stock Prices (5 Years)", fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend()
ax.annotate("Simulated data using random walk (GBM)",
            xy=(0.01, 0.02), xycoords='axes fraction', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig("plots/01_stock_prices.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 01_stock_prices.png")

# ----- Plot 2: Cumulative Returns of Each Strategy -----
cumulative = (1 + all_returns).cumprod()

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#2196F3', '#FF9800', '#4CAF50']
for i, col in enumerate(cumulative.columns):
    ax.plot(cumulative.index, cumulative[col], label=col, linewidth=2, color=colors[i])

ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Starting value')
ax.set_title("Cumulative Returns: Strategy Comparison", fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1 Invested")
ax.legend()

# Add final value labels at the end of each line
for i, col in enumerate(cumulative.columns):
    final_val = cumulative[col].iloc[-1]
    ax.annotate(f"${final_val:.2f}", xy=(cumulative.index[-1], final_val),
                xytext=(5, 0), textcoords='offset points',
                fontsize=9, color=colors[i], fontweight='bold')

plt.tight_layout()
plt.savefig("plots/02_cumulative_returns.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 02_cumulative_returns.png")

# ----- Plot 3: Drawdown Charts -----
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle("Drawdown Analysis by Strategy", fontsize=15, fontweight='bold', y=1.01)

for i, col in enumerate(all_returns.columns):
    cum = (1 + all_returns[col]).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max

    axes[i].fill_between(drawdown.index, drawdown.values.astype(float), 0,
                         color=colors[i], alpha=0.4, label='Drawdown')
    axes[i].plot(drawdown.index, drawdown.values.astype(float), color=colors[i], linewidth=0.8)
    axes[i].set_ylabel("Drawdown")
    axes[i].set_title(f"{col}  |  Max Drawdown: {drawdown.min():.1%}", fontsize=11)
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("plots/03_drawdown_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 03_drawdown_analysis.png")

# ----- Plot 4: Correlation Heatmap -----
corr = all_returns.corr()

fig, ax = plt.subplots(figsize=(7, 5))
mask = np.zeros_like(corr, dtype=bool)  # show all cells
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 13, "weight": "bold"})
ax.set_title("Strategy Return Correlation Heatmap\n(1 = perfectly correlated, 0 = no correlation)",
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("plots/04_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 04_correlation_heatmap.png")

# ----- Plot 5: Sharpe Ratio Bar Chart -----
sharpe_vals = {col: sharpe_ratio(all_returns[col]) for col in all_returns.columns}

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(sharpe_vals.keys(), sharpe_vals.values(), color=colors, width=0.5,
              edgecolor='white', linewidth=1.2)

# Add value labels on top of bars
for bar, val in zip(bars, sharpe_vals.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f"{val:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Sharpe = 1 (good)')
ax.axhline(y=0, color='red',   linestyle='--', alpha=0.5, label='Sharpe = 0 (break even)')
ax.set_title("Sharpe Ratio Comparison\n(Higher is better — measures return per unit of risk)",
             fontsize=12, fontweight='bold', pad=12)
ax.set_ylabel("Sharpe Ratio")
ax.legend(fontsize=9)
ax.set_ylim(bottom=min(0, min(sharpe_vals.values()) - 0.3))
plt.tight_layout()
plt.savefig("plots/05_sharpe_ratio.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 05_sharpe_ratio.png")

# ----- Plot 6: Full Dashboard (Summary) -----
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Quantitative Strategy Evaluation Dashboard", fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Top row: cumulative returns (spans 2 columns)
ax1 = fig.add_subplot(gs[0, :2])
for i, col in enumerate(cumulative.columns):
    ax1.plot(cumulative.index, cumulative[col], label=col, linewidth=2, color=colors[i])
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.4)
ax1.set_title("Cumulative Returns", fontweight='bold')
ax1.set_ylabel("Growth of $1")
ax1.legend(fontsize=8)

# Top right: Sharpe bar
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(range(len(sharpe_vals)), list(sharpe_vals.values()), color=colors)
ax2.set_xticks(range(len(sharpe_vals)))
ax2.set_xticklabels(['B&H', 'Mom.', 'SMA'], fontsize=9)
ax2.axhline(y=1, color='green', linestyle='--', alpha=0.7)
ax2.set_title("Sharpe Ratio", fontweight='bold')
for j, v in enumerate(sharpe_vals.values()):
    ax2.text(j, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Middle: drawdowns
for i, col in enumerate(all_returns.columns):
    ax = fig.add_subplot(gs[1, i])
    cum = (1 + all_returns[col]).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()
    ax.fill_between(dd.index, dd.values.astype(float), 0, color=colors[i], alpha=0.4)
    ax.plot(dd.index, dd.values, color=colors[i], linewidth=0.8)
    ax.set_title(f"{col}\nMax DD: {dd.min():.1%}", fontsize=9, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(axis='x', labelsize=6)

# Bottom left: metrics table
ax_table = fig.add_subplot(gs[2, :2])
ax_table.axis('off')
table_data = []
for col in all_returns.columns:
    r = all_returns[col]
    table_data.append([col,
                        f"{total_return(r):.1%}",
                        f"{annualized_return(r):.1%}",
                        f"{volatility(r):.1%}",
                        f"{sharpe_ratio(r):.2f}",
                        f"{max_drawdown(r):.1%}"])

col_labels = ['Strategy', 'Total Return', 'Annual Return', 'Volatility', 'Sharpe', 'Max DD']
tbl = ax_table.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 1.6)
for (row, col_), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#37474F')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#ECEFF1')
ax_table.set_title("Performance Summary Table", fontweight='bold', pad=8)

# Bottom right: correlation heatmap
ax_corr = fig.add_subplot(gs[2, 2])
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.5, ax=ax_corr,
            annot_kws={"size": 9},
            cbar_kws={"shrink": 0.8})
ax_corr.set_title("Correlation", fontweight='bold')
ax_corr.tick_params(axis='x', rotation=30, labelsize=7)
ax_corr.tick_params(axis='y', rotation=0,  labelsize=7)

plt.savefig("plots/06_full_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 06_full_dashboard.png")

# -------------------------------------------------------
# STEP 6: Print Final Summary
# -------------------------------------------------------
print("\n[6/6] Final Summary")
print("-" * 45)
print(metrics_df.to_string())
print("\nAll plots saved to the 'plots/' folder.")
print("Data saved to the 'data/' folder.")
print("\nDone! ✓")
