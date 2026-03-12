# 📈 Quantitative Strategy Evaluation and Comparison

*A beginner data science portfolio project*

---

## 🎯 What This Project Does

This project compares **3 simple stock trading strategies** to see which one performs best over a 5-year period. Instead of using real market data (which requires an API key), I simulate stock prices using a technique called a **random walk** — the same mathematical model used in professional finance.

The three strategies I compare are:

| Strategy | Simple Explanation |
|----------|-------------------|
| **Buy & Hold** | Buy the stock on day 1 and never sell — just hold it |
| **Momentum** | Buy when the stock has been going up recently |
| **SMA Crossover** | Buy/sell based on short vs. long-term moving averages |

---

## 📁 Project Structure

```
quant_project/
│
├── notebook.ipynb          ← Main Jupyter notebook with all the code
├── generate_plots.py       ← Standalone Python script (same logic)
├── README.md               ← This file
│
├── data/
│   ├── simulated_prices.csv     ← Simulated stock prices
│   └── performance_metrics.csv  ← Strategy comparison results
│
└── plots/
    ├── 01_stock_prices.png       ← Simulated stock price chart
    ├── 02_cumulative_returns.png ← How each strategy grows $1
    ├── 03_drawdown_analysis.png  ← Worst losses from each strategy
    ├── 04_correlation_heatmap.png← How strategies relate to each other
    ├── 05_sharpe_ratio.png       ← Risk-adjusted return comparison
    └── 06_full_dashboard.png     ← All-in-one summary chart
```

---

## 📊 How the Strategies Are Compared

I use 5 standard performance metrics from quantitative finance:

**1. Total Return**
How much money you made over the whole period.
Example: 100% means your $1,000 became $2,000.

**2. Annual Return**
The average yearly return. More useful than total return for comparing strategies of different lengths.

**3. Volatility**
How much the daily returns bounce around. Higher volatility = more risk.

**4. Sharpe Ratio**
The most important metric. It measures *return per unit of risk*:
```
Sharpe = (Strategy Return - Risk Free Rate) / Volatility
```
A Sharpe above 1.0 is generally considered good. Higher is better!

**5. Maximum Drawdown**
The biggest loss from a peak to a valley. If your portfolio hit $150 then fell to $105, that's a -30% drawdown. This tells you the worst case you would have experienced.

---

## 📈 What the Results Show

Here's a summary of my results (your numbers may vary if you use different random seeds):

| Strategy | Total Return | Sharpe Ratio | Max Drawdown |
|----------|-------------|--------------|--------------|
| Buy & Hold | ~194% | ~1.09 | ~-17% |
| Momentum | ~97% | ~0.82 | ~-22% |
| SMA Crossover | ~58% | ~0.54 | ~-25% |

**Key findings:**

- **Buy & Hold won** on total return. This is actually very common in finance — it's hard to beat just holding a good stock!
- **Momentum and SMA strategies** reduce volatility slightly by sitting out during bad periods, but they also miss good days.
- The **correlation heatmap** shows that all 3 strategies are somewhat correlated because they're all based on the same underlying stock.
- The SMA strategy has the **lowest Sharpe ratio**, meaning it takes more risk per unit of return.

> ⚠️ **Important note**: These are simulated results with no transaction costs. In real trading, buying and selling frequently adds costs that can significantly reduce returns.

---

## 🚀 How to Run This Project

### Option 1: Google Colab (Easiest — no installation needed)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `notebook.ipynb` from this folder
4. Click **Runtime → Run all** (or press `Ctrl+F9`)
5. All charts will appear inline in the notebook!

> 💡 Google Colab already has `pandas`, `numpy`, `matplotlib`, and `seaborn` installed — no setup needed!

### Option 2: Run Locally

**Requirements:** Python 3.8+

1. Install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn
```

2. Run the notebook:
```bash
jupyter notebook notebook.ipynb
```

3. Or run the standalone script:
```bash
python generate_plots.py
```

Charts will be saved to the `plots/` folder.

---

## 🧰 Libraries Used

| Library | What it's used for |
|---------|-------------------|
| `numpy` | Math and random number generation |
| `pandas` | Data tables and time series |
| `matplotlib` | Making charts and plots |
| `seaborn` | Prettier heatmaps and styling |

No external data APIs needed — everything is simulated!

---

## 💡 What I Learned

- How to simulate financial time series data
- How to implement basic trading strategies using pandas
- The importance of avoiding **look-ahead bias** (never use future info to make today's decision)
- How to interpret Sharpe ratio, drawdowns, and correlation
- Why simple strategies often struggle to beat Buy & Hold

## 🔮 Future Improvements

- [ ] Add real stock data using `yfinance`
- [ ] Include transaction costs in the backtest
- [ ] Add more strategies (RSI, Bollinger Bands)
- [ ] Test across multiple stocks (portfolio)
- [ ] Add statistical significance testing

---

*Created as a beginner data science portfolio project. All data is simulated — this is not financial advice!*
