<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../logo-dark.svg" width="120">
  <source media="(prefers-color-scheme: light)" srcset="../logo.svg" width="120">
  <img alt="SignalFlow" src="../logo.png" width="120">
</picture>

# sf-kedro

**SignalFlow Universal Pipelines — Kedro-based backtesting, optimization, and validation**

<p>
<a href="#"><img src="https://img.shields.io/badge/version-0.5.0-7c3aed" alt="Version"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-3b82f6?logo=python&logoColor=white" alt="Python 3.12+"></a>
<img src="https://img.shields.io/badge/kedro-1.1-f59e0b" alt="Kedro">
<img src="https://img.shields.io/badge/optuna-3b82f6" alt="Optuna">
</p>

</div>

---

Part of the [SignalFlow](https://github.com/pathway2nothing/sf-project) ecosystem.

Implements **Universal Pipelines Architecture** — pipelines defined by purpose, not by strategy name. Flow configuration is modular and passed via parameters.

```
FLOW CONFIG (conf/base/flows/*.yml)
├── detector   (required) → signal generation
├── validator  (optional) → ML signal filtering
└── strategy   (optional) → entry/exit rules

UNIVERSAL PIPELINES
├── backtest   → run backtest for any flow
├── analyze    → analyze features and signals
├── train      → train validator model
├── tune       → Optuna parameter optimization
└── validate   → walk-forward validation
```

## Installation

```bash
conda create --name sf-kedro python==3.12
conda activate sf-kedro
pip install -r requirements.txt
cp .env.example .env
```

## Quick Start

```bash
kedro run --pipeline=backtest --params='flow_id=grid_sma'
kedro run --pipeline=analyze --params='flow_id=grid_sma'
kedro run --pipeline=tune --params='flow_id=grid_sma,n_trials=100'
kedro run --pipeline=validate --params='flow_id=grid_sma,n_folds=5'
kedro run --pipeline=train --params='flow_id=grid_sma'
```

## Pipelines

### backtest

Run backtest for any flow configuration.

**Nodes:** load_flow_data → run_flow_detection → run_flow_backtest → compute_metrics → save_flow_plots

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
==================================================
```

### analyze

Feature exploration and signal quality analysis.

```bash
kedro run --pipeline=analyze --params='flow_id=grid_sma,level=signals'
```

Levels: `features`, `signals`, `all`

### train

Train ML validator for signal filtering.

**Nodes:** load_training_data → prepare_features → train_validator → save_model

### tune

Optuna hyperparameter optimization.

```bash
kedro run --pipeline=tune --params='flow_id=grid_sma,n_trials=100,level=strategy'
```

Levels: `detector`, `strategy`

### validate

Walk-forward out-of-sample validation.

```bash
kedro run --pipeline=validate --params='flow_id=grid_sma,n_folds=5'
```

```
==================================================
Walk-Forward Validation: Grid SMA Crossover
--------------------------------------------------
  Valid folds:     5/5
  Avg Return:      +1.23%
  Total trades:    1250
==================================================
```

## Flow Configuration

```yaml
# conf/base/flows/grid_sma.yml
flow_id: grid_sma
flow_name: "Grid SMA Crossover"

data:
  pairs: [BTCUSDT, ETHUSDT]

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

## Project Structure

```
sf-kedro/
├── conf/base/
│   ├── parameters/          # Pipeline-specific params
│   │   ├── common.yml       # Shared defaults
│   │   ├── backtest.yml
│   │   ├── analyze.yml
│   │   ├── train.yml
│   │   ├── tune.yml
│   │   └── validate.yml
│   ├── flows/               # Flow configs
│   │   └── grid_sma.yml
│   └── catalog/             # Data catalog
├── src/sf_kedro/
│   ├── pipelines/
│   │   ├── backtest/
│   │   ├── analyze/
│   │   ├── train/
│   │   ├── tune/
│   │   └── validate/
│   └── utils/
│       ├── flow_config.py
│       ├── detection.py
│       └── telegram.py
└── data/
```

## Integrations

| Integration | Purpose |
|-------------|---------|
| **MLflow / DagsHub** | Experiment tracking, model registry |
| **Optuna** | Hyperparameter optimization |
| **Telegram** | Automated notifications |
| **Plotly** | Interactive visualizations |

## Dependencies

signalflow-trading, signalflow-ta, signalflow-nn, Kedro, Polars, Optuna, Plotly, pyTelegramBotAPI

---

**License:** Proprietary &ensp;·&ensp; Part of [SignalFlow](https://github.com/pathway2nothing/sf-project)
