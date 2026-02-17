# SF Kedro

**SignalFlow Universal Pipelines** — Kedro-based framework for trading strategy backtesting, optimization, and validation.

Part of the [SignalFlow](https://signalflow-trading.com) ecosystem.

## Overview

`sf-kedro` implements **Universal Pipelines Architecture** — pipelines defined by **purpose**, not by strategy name. Flow configuration is modular and passed via parameters.

```
FLOW CONFIG (conf/base/flows/*.yml)
├── detector   (required) → signal generation
├── validator  (optional) → ML signal filtering
└── strategy   (optional) → entry/exit rules for backtest

UNIVERSAL PIPELINES (by purpose)
├── backtest   → Run backtest for any flow
├── analyze    → Analyze features and signals
├── train      → Train validator model
├── tune       → Optuna parameter optimization
└── validate   → Walk-forward validation
```

## Installation

```bash
# Create environment
conda create --name sf-kedro python==3.12
conda activate sf-kedro

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

Required environment variables:
- `TELEGRAM_BOT_TOKEN` — Telegram bot token (optional)
- `TELEGRAM_CHAT_ID` — Telegram chat ID (optional)

## Quick Start

```bash
# Run backtest
kedro run --pipeline=backtest --params='flow_id=grid_sma'

# Analyze signals
kedro run --pipeline=analyze --params='flow_id=grid_sma'

# Optimize parameters
kedro run --pipeline=tune --params='flow_id=grid_sma,n_trials=100'

# Walk-forward validation
kedro run --pipeline=validate --params='flow_id=grid_sma,n_folds=5'

# Train validator
kedro run --pipeline=train --params='flow_id=grid_sma'

# List available flows
python -c "from sf_kedro.utils.flow_config import list_flows; print(list_flows())"
```

## Project Structure

```
sf-kedro/
├── conf/
│   └── base/
│       ├── parameters/
│       │   ├── common.yml          # Shared defaults
│       │   ├── backtest.yml        # Pipeline-specific params
│       │   ├── analyze.yml
│       │   ├── train.yml
│       │   ├── tune.yml
│       │   └── validate.yml
│       ├── flows/
│       │   └── grid_sma.yml        # Flow: Grid + SMA detector
│       └── catalog/
│           └── *.yml               # Data catalog definitions
├── src/sf_kedro/
│   ├── pipelines/
│   │   ├── backtest/               # Backtest pipeline
│   │   ├── analyze/                # Analysis pipeline
│   │   ├── train/                  # Validator training
│   │   ├── tune/                   # Optuna optimization
│   │   └── validate/               # Walk-forward validation
│   └── utils/
│       ├── flow_config.py          # Flow config loader
│       ├── detection.py            # Detection utilities
│       └── telegram.py             # Telegram notifications
└── data/                           # Local data storage
```

## Pipelines

### 1. Backtest

Run backtest for any flow configuration.

```bash
kedro run --pipeline=backtest --params='flow_id=grid_sma'
```

**Nodes**: load_flow_data → run_flow_detection → run_flow_backtest → compute_metrics → save_flow_plots

**Output**:
```
==================================================
Backtest Complete: Grid SMA Crossover
--------------------------------------------------
  Initial Capital: $10,000.00
  Final Equity:    $9,662.57
  Total Return:    -3.37%
  Trades Executed: 756
  Win Rate:        34.6%
  Max Drawdown:    3.66%
  Sharpe Ratio:    0.01
==================================================
```

### 2. Analyze

Analyze features and signal quality.

```bash
kedro run --pipeline=analyze --params='flow_id=grid_sma'
kedro run --pipeline=analyze --params='flow_id=grid_sma,level=signals'
```

**Levels**: `features`, `signals`, `all`

**Output**: Feature statistics, signal distribution, correlation analysis.

### 3. Train

Train ML validator for signal filtering.

```bash
kedro run --pipeline=train --params='flow_id=grid_sma'
```

**Nodes**: load_training_data → prepare_features → train_validator → save_model

**Output**: Trained validator model saved to `data/06_models/`.

### 4. Tune

Optimize parameters using Optuna.

```bash
kedro run --pipeline=tune --params='flow_id=grid_sma,n_trials=100'
kedro run --pipeline=tune --params='flow_id=grid_sma,level=strategy'
```

**Levels**: `detector`, `strategy`

**Output**: Best parameters saved to `data/06_models/best_params_*.yml`.

### 5. Validate

Walk-forward validation for out-of-sample testing.

```bash
kedro run --pipeline=validate --params='flow_id=grid_sma,n_folds=5'
```

**Nodes**: load_validation_data → run_walk_forward → save_validation_report

**Output**:
```
==================================================
Walk-Forward Validation: Grid SMA Crossover
--------------------------------------------------
  Valid folds:     5/5
  Avg Return:      +1.23%
  Total trades:    1250
  Per-fold results:
    Fold 1: +0.85% (245 trades)
    Fold 2: +1.12% (267 trades)
    ...
==================================================
```

## Flow Configuration

### Example: Grid SMA Flow

```yaml
# conf/base/flows/grid_sma.yml

flow_id: grid_sma
flow_name: "Grid SMA Crossover"

data:
  pairs:
    - BTCUSDT
    - ETHUSDT

detector:
  type: "example/sma_cross"
  fast_period: 60
  slow_period: 720

strategy:
  entry_rules:
    - type: "signal"
      base_position_size: 200.0
      max_positions_per_pair: 5
      entry_filters:
        - type: "price_distance_filter"
          min_distance_pct: 0.02

  exit_rules:
    - type: "tp_sl"
      take_profit_pct: 0.015
      stop_loss_pct: 0.01

  metrics:
    - type: "total_return"
    - type: "win_rate"
    - type: "sharpe_ratio"
    - type: "drawdown"
    - type: "profit_factor"
```

### Common Config

Shared defaults in `conf/base/parameters/common.yml`:

```yaml
telegram:
  enabled: false
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"

data:
  store:
    db_path: "data/01_raw/market.duckdb"
  period:
    start: { year: 2024, month: 1, day: 1 }
    end: { year: 2025, month: 1, day: 1 }
```

## Telegram Notifications

Enable notifications in flow config:

```yaml
telegram:
  enabled: true
```

Set environment variables:
```bash
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

## Development

```bash
# Run tests
pytest tests/

# Pipeline visualization
kedro viz

# Jupyter notebooks
kedro jupyter notebook
```

## Dependencies

- **Kedro** — Pipeline orchestration
- **signalflow-trading** — Core infrastructure
- **signalflow-ta** — Technical indicators
- **signalflow-nn** — Neural network models
- **Polars** — Data processing
- **Optuna** — Hyperparameter optimization
- **Plotly** — Visualization
- **pyTelegramBotAPI** — Telegram integration

## Links

- **Documentation**: https://signalflow-trading.com
- **Core Library**: [signalflow-trading](https://github.com/sf-project/signalflow-trading)
- **Technical Analysis**: [signalflow-ta](https://github.com/sf-project/signalflow-ta)
- **Neural Networks**: [signalflow-nn](https://github.com/sf-project/signalflow-nn)

## License

Proprietary — Part of the SignalFlow project.
