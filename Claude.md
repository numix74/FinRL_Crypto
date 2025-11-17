# Claude.md - FinRL-Crypto Project Guide

## Project Overview

**FinRL-Crypto** is a Deep Reinforcement Learning (DRL) framework for cryptocurrency trading that addresses the critical problem of backtest overfitting. The project implements three validation methodologies (CPCV, K-Fold CV, Walk-Forward) to improve model generalization in live trading.

**Key Paper**: [Deep reinforcement learning for cryptocurrency trading: Practical approach to address backtest overfitting](https://arxiv.org/abs/2209.05559) (AAAI '23)

**Main Achievement**: Reduces overfitting in DRL models by 46% compared to traditional methods.

## Core Concepts

### Validation Methods
1. **CPCV (Combinatorial Purged Cross-Validation)**: Advanced cross-validation that prevents data leakage between temporally correlated financial data
2. **K-Fold Cross-Validation**: Standard k-fold approach adapted for time-series financial data
3. **Walk-Forward Validation**: Rolling window approach simulating real-world trading conditions

### Key Metrics
- **PBO (Probability of Backtest Overfitting)**: Quantifies the likelihood that observed backtest performance is due to overfitting rather than genuine strategy efficacy

## Directory Structure

```
FinRL_Crypto/
├── data/                       # Training/validation data storage
│   └── trade_data/            # Downloaded trading data (created after running 0_dl_*.py)
├── drl_agents/                # ElegantRL framework implementation
│   ├── agents/                # DRL algorithm implementations
│   └── models/                # Neural network architectures
├── plots_and_metrics/         # Analysis outputs (plots, metrics)
├── train/                     # Training utility functions
├── train_results/             # Saved trained DRL agents (auto-generated)
└── [numbered Python files]    # Main workflow scripts (see below)
```

## Key Files

### Configuration
- **`config_main.py`**: Central configuration file
  - Training/validation data sizes (`no_candles_for_train`, `no_candles_for_val`)
  - Trading period (`trade_start_date`, `trade_end_date`)
  - Cryptocurrency tickers (`TICKER_LIST`)
  - Technical indicators (`TECHNICAL_INDICATORS_LIST`)
  - Cross-validation parameters (`KCV_groups`, `K_TEST_GROUPS`, `NUM_PATHS`)
  - Auto-calculates train/val dates based on candle counts

- **`config_api.py`**: API credentials (Binance, etc.) - keep sensitive data here

### Data Processing
- **`processor_Binance.py`**: Downloads and processes data from Binance API
- **`processor_Yahoo.py`**: Alternative data source (Yahoo Finance)
- **`processor_Base.py`**: Base class for data processors

### Environment
- **`environment_Alpaca.py`**: Custom OpenAI Gym environment for crypto trading
  - State space: Technical indicators for all tickers
  - Action space: Portfolio allocation decisions
  - Reward: Portfolio returns with transaction costs

### Core Functions
- **`function_CPCV.py`**: Implements Combinatorial Purged Cross-Validation logic
- **`function_PBO.py`**: Computes Probability of Backtest Overfitting
- **`function_finance_metrics.py`**: Financial performance metrics (Sharpe, Sortino, Max Drawdown, etc.)
- **`function_train_test.py`**: Training and testing utilities for DRL agents

## Workflow (Numbered Scripts)

Execute scripts in numerical order:

### 0. Data Download
```bash
python 0_dl_trainval_data.py   # Download training & validation data
python 0_dl_trade_data.py      # Download trading (test) data
```

### 1. Hyperparameter Optimization (Choose ONE)
```bash
python 1_optimize_cpcv.py      # CPCV method (recommended for preventing overfitting)
python 1_optimize_kcv.py       # K-Fold CV method
python 1_optimize_wf.py        # Walk-Forward method
```
- Uses Optuna for hyperparameter optimization
- Results saved to `train_results/[timestamp]/`
- Trials: Controlled by `H_TRIALS` in config

### 2. Validation Analysis
```bash
python 2_validate.py
```
- Analyzes training/validation process
- Specify results folder from `train_results/`
- Generates plots in `plots_and_metrics/`

### 4. Backtesting
```bash
python 4_backtest.py
```
- Tests trained agents on held-out trading data
- Input: List of result folders from `train_results/`
- Outputs: Performance metrics, equity curves, trade logs

### 5. PBO Analysis
```bash
python 5_pbo.py
```
- Computes Probability of Backtest Overfitting
- Input: Multiple result folders from `train_results/`
- Critical for assessing strategy robustness

## Common Tasks

### Changing Trading Assets
Edit `config_main.py`:
```python
TICKER_LIST = ['BTCUSDT', 'ETHUSDT', ...]
ALPACA_LIMITS = np.array([0.0001, 0.001, ...])  # Min buy limits (same order)
```

### Adjusting Training Period
Edit `config_main.py`:
```python
no_candles_for_train = 20000  # Number of candles for training
no_candles_for_val = 5000     # Number of candles for validation
trade_start_date = '2022-04-30 00:00:00'
trade_end_date = '2022-06-27 00:00:00'
```
Training/validation dates are auto-calculated backward from `trade_start_date`.

### Modifying Technical Indicators
Edit `config_main.py`:
```python
TECHNICAL_INDICATORS_LIST = ['open', 'high', 'low', 'close', 'volume',
                             'macd', 'rsi', 'cci', 'dx', ...]
```
Ensure indicators are implemented in processor files.

### Changing DRL Algorithm
Navigate to `drl_agents/` and select from available algorithms (PPO, A2C, SAC, TD3, etc.). Modify imports in optimization scripts.

## Technical Details

### State Space
- Multi-asset POMDP (Partially Observable Markov Decision Process)
- State vector: `[balance, holdings[], prices[], technical_indicators[][]]`
- Dimension: `1 + n_tickers + n_tickers * (1 + n_indicators)`

### Action Space
- Continuous: Portfolio weights for each asset [-1, 1]
- -1: Short (if enabled), 0: Hold, +1: Long
- Actions normalized to sum to 1 (or cash allocation)

### Reward Function
Located in `environment_Alpaca.py`:
- Primary: Portfolio return (percent change)
- Penalties: Transaction costs, minimum trade limits
- Customizable for risk-adjusted returns (Sharpe, etc.)

### Data Considerations
- **Timeframe**: Default `5m` candles (configurable: 1m, 5m, 10m, 30m, 1h, 2h, 4h, 12h)
- **Lookback**: Ensure sufficient historical data for technical indicators (e.g., 200-period moving averages)
- **Binance Limits**: API rate limits apply; implement exponential backoff if needed

## Development Guidelines

### Adding New Validation Methods
1. Create new file: `1_optimize_[method_name].py`
2. Implement train/val split logic
3. Integrate with Optuna hyperparameter search
4. Save results to `train_results/[method_name]_[timestamp]/`

### Adding New Metrics
Edit `function_finance_metrics.py`:
- Add metric calculation function
- Follow existing format (takes returns/prices as input)
- Update analysis scripts to display new metric

### Debugging Training Issues
- Check `config_main.py` dates and candle counts
- Verify data downloaded correctly in `data/` and `data/trade_data/`
- Inspect `train_results/[folder]/logs/` for training logs
- Reduce `H_TRIALS` for faster iteration during debugging

### Common Pitfalls
1. **Insufficient Data**: Ensure `no_candles_for_train` + `no_candles_for_val` doesn't exceed available historical data
2. **Date Alignment**: Trading period must be AFTER training/validation period
3. **API Keys**: Set credentials in `config_api.py` before running download scripts
4. **Memory Issues**: Large `TICKER_LIST` or long training periods may require GPU/high-RAM machines

## Dependencies

Key libraries (see `requirements.txt`):
- `elegantrl`: DRL framework
- `gym`: Environment interface
- `optuna`: Hyperparameter optimization
- `pandas`, `numpy`: Data manipulation
- `ccxt` / `python-binance`: Exchange connectivity
- `ta` / `stockstats`: Technical indicators

## Citation

If using this codebase:
```bibtex
@article{gort2022deep,
  title={Deep reinforcement learning for cryptocurrency trading: Practical approach to address backtest overfitting},
  author={Gort, Berend Jelmer Dirk and Liu, Xiao-Yang and Gao, Jiechao and Chen, Shuaiyu and Wang, Christina Dan},
  journal={AAAI Bridge on AI for Financial Services},
  year={2023}
}
```

## Useful Commands

### Quick Start
```bash
# 1. Configure settings
vim config_main.py

# 2. Download data
python 0_dl_trainval_data.py && python 0_dl_trade_data.py

# 3. Train with CPCV (recommended)
python 1_optimize_cpcv.py

# 4. Backtest
python 4_backtest.py  # Edit script to specify train_results folder

# 5. Assess overfitting
python 5_pbo.py  # Edit script to specify train_results folders
```

### Analyzing Results
Results are in `train_results/[timestamp]/`:
- `best_params.json`: Optimal hyperparameters found
- `agents/`: Saved model checkpoints
- `logs/`: Training logs and metrics
- `performance.csv`: Trial-by-trial performance

## Architecture Notes

### ElegantRL Framework
- Located in `drl_agents/`
- Supports: PPO, A2C, TD3, SAC, DDPG, etc.
- Efficient parallel training with vectorized environments
- GPU acceleration support

### Cross-Validation Strategy
- **Purging**: Removes data points temporally close to test set to prevent leakage
- **Embargo**: Additional time gap between train and test
- **Combinatorial**: Tests all possible train/test splits for robust validation

## Questions & Support

- Original Paper: https://arxiv.org/abs/2209.05559
- Author: [Berend Gort](https://www.linkedin.com/in/bjdg/)
- Community: AI4Finance community

---

*This file is intended for AI assistants (like Claude) to quickly understand the codebase structure, purpose, and usage patterns for effective collaboration on this project.*
