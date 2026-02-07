# SF Kedro

**SignalFlow Applied ML Pipelines** — Kedro-based experimentation and research framework for quantitative trading strategies.

Part of the [SignalFlow](https://signalflow-trading.com) ecosystem for technical analysis and ML-driven trading.

## Overview

`sf-kedro` is the applied ML repository in the SignalFlow project. It provides production-ready Kedro pipelines for:
- **Signal detection & backtesting** (baseline strategies)
- **Feature analysis** (statistical analysis, correlation, distribution plots)
- **ML model validation** (classical ML approaches)
- **Neural network experiments** (deep learning models)
- **Production deployment** (final strategies)

All pipelines integrate with **MLflow/DagsHub** for experiment tracking and **Telegram** for notifications.

## Installation

### 1. Create conda environment

```bash
conda create --name sf-kedro python==3.12
conda activate sf-kedro
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required variables:
- `MLFLOW_ENABLED` — Enable/disable MLflow tracking
- `DAGSHUB_REPO` — Your DagsHub repository (format: `username/repo-name`)
- `MLFLOW_TRACKING_URI` — MLflow tracking URI
- `TELEGRAM_BOT_TOKEN` — Telegram bot token (optional, for notifications)
- `TELEGRAM_CHAT_ID` — Telegram chat ID (optional)

### 4. Configure BigQuery credentials

Add `conf/local/credentials.yml` (gitignored):

```yaml
gcp_credentials:
  project_id: your-project-id
  credentials_path: /path/to/service-account.json
```

## Project Structure

```
sf-kedro/
├── conf/
│   ├── base/                    # Base configuration
│   │   ├── catalog/             # Data catalog definitions
│   │   ├── parameters/          # Pipeline parameters
│   │   │   ├── baseline.yml
│   │   │   ├── feature_analysis.yml
│   │   │   ├── ml_validated.yml
│   │   │   └── nn_validated.yml
│   │   └── parameters.yml       # Global parameters
│   └── local/                   # Local overrides (gitignored)
│       └── credentials.yml
├── src/sf_kedro/
│   ├── general_nodes/           # Reusable pipeline nodes
│   ├── hooks/                   # Kedro hooks (DagsHub, Telegram)
│   ├── pipelines/               # Pipeline definitions
│   │   ├── baseline/            # Baseline strategy experiments
│   │   ├── feature_analysis/    # Feature analysis & visualization
│   │   ├── ml_validated/        # ML model validation
│   │   ├── nn_validated/        # Neural network experiments
│   │   └── production/          # Production-ready pipelines
│   └── utils/                   # Helper utilities
└── data/                        # Local data storage (gitignored)
```

## Pipelines

### 1. Baseline (`baseline`)

**Purpose**: Test baseline trading strategies with signal detection and backtesting.

**Steps**:
- Download market data from BigQuery
- Detect trading signals (technical indicators)
- Generate labels (forward returns, outcomes)
- Compute signal metrics (distribution, profitability)
- Run backtest with strategy logic
- Generate performance plots
- Send Telegram notifications

**Run**:
```bash
kedro run --pipeline=baseline
```

**Config**: `conf/base/parameters/baseline.yml`

---

### 2. Feature Analysis (`feature_analysis`)

**Purpose**: Analyze feature distributions, correlations, and statistical properties.

**Steps**:
- Download market data
- Extract features (technical indicators, price patterns)
- Build analysis plots:
  - Feature distributions
  - Correlation matrices
  - Time series evolution
  - Cross-pair comparisons
- Send results to Telegram

**Run**:
```bash
kedro run --pipeline=feature_analysis
```

**Config**: `conf/base/parameters/feature_analysis.yml`

---

### 3. ML Validated (`ml_validated`)

**Purpose**: Train and validate classical ML models (Random Forest, XGBoost, etc.)

**Steps**:
- Data loading & preprocessing
- Feature engineering
- Model training with cross-validation
- Performance evaluation
- MLflow experiment tracking

**Run**:
```bash
kedro run --pipeline=ml_validated
```

**Config**: `conf/base/parameters/ml_validated.yml`

---

### 4. NN Validated (`nn_validated`)

**Purpose**: Experiment with neural network models (PyTorch-based).

**Steps**:
- Data loading & preprocessing
- Sequence preparation for time series
- Neural network training (LSTM, Transformers, etc.)
- Validation and testing
- MLflow tracking

**Run**:
```bash
kedro run --pipeline=nn_validated
```

**Config**: `conf/base/parameters/nn_validated.yml`

---

### 5. Production (`production`)

**Purpose**: Production-ready pipelines for deployment.

**Run**:
```bash
kedro run --pipeline=production
```

## Configuration

### Parameters Structure

Global parameters in `conf/base/parameters.yml`:
- `telegram` — Telegram bot configuration
- `strategy_name` — Strategy identifier

Pipeline-specific parameters in `conf/base/parameters/<pipeline>.yml`:
- `data` — Data loading config (pairs, period, store)
- `detector` — Signal detection parameters
- `labeling` — Label generation config
- `strategy` — Backtesting strategy parameters
- `features` — Feature extraction config (for feature_analysis)

### Example: Baseline Parameters

```yaml
baseline:
  data:
    store: bigquery
    loader: OHLCVLoader
    period: 1h
    pairs: ["BTC/USDT", "ETH/USDT"]

  detector:
    name: "overlap/sma_cross"
    short_period: 10
    long_period: 30

  strategy:
    initial_balance: 10000
    position_size: 0.1
```

## Running Pipelines

### Run entire pipeline
```bash
kedro run --pipeline=baseline
```

### Run specific node
```bash
kedro run --node=detect_signals_node
```

### Run with tags
```bash
kedro run --tag=data_download
kedro run --tag=backtesting
```

### Skip specific tags
```bash
kedro run --pipeline=baseline --skip-tag=data_download
```

## Integrations

### MLflow / DagsHub

Experiment tracking is configured via hooks in `src/sf_kedro/hooks/dagshub_hooks.py`.

Enable/disable in `.env`:
```bash
MLFLOW_ENABLED=true
DAGSHUB_REPO=username/sf-kedro
MLFLOW_TRACKING_URI=https://dagshub.com/username/sf-kedro.mlflow
```

Logged automatically:
- Pipeline parameters
- Model metrics
- Backtest results
- Generated plots

### Telegram Notifications

Send pipeline results to Telegram (plots, metrics, summaries).

Configure in `conf/base/parameters.yml`:
```yaml
telegram:
  enabled: true
  bot_token: null  # Uses TELEGRAM_BOT_TOKEN env var
  chat_id: null    # Uses TELEGRAM_CHAT_ID env var
  image_width: 1400
  image_height: 900
  send_text_report: false
```

Set environment variables:
```bash
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

## Development

### Run tests
```bash
pytest tests/
```

### Kedro viz (pipeline visualization)
```bash
kedro viz
```

### Jupyter notebooks
```bash
kedro jupyter notebook
```

## Dependencies

Core libraries:
- **Kedro** `~=1.1.1` — Pipeline orchestration
- **signalflow-trading** `>=0.3.7` — Core infrastructure
- **signalflow-ta** `>=0.3.6` — Technical indicators (200+)
- **signalflow-nn** `>=0.2.6` — Neural network models
- **MLflow** — Experiment tracking
- **DagsHub** — MLflow backend
- **Plotly** — Visualization
- **pyTelegramBotAPI** — Telegram integration

See `requirements.txt` for full list.

## Links

- **Documentation**: https://signalflow-trading.com
- **GitHub Organization**: https://github.com/sf-project
- **Core Library**: [signalflow-trading](https://github.com/sf-project/signalflow-trading)
- **Technical Analysis**: [signalflow-ta](https://github.com/sf-project/signalflow-ta)
- **Neural Networks**: [signalflow-nn](https://github.com/sf-project/signalflow-nn)

## License

Proprietary — Part of the SignalFlow project.
