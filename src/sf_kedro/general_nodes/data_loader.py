"""Data loading and storage operations."""

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import mlflow

import signalflow as sf


def download_market_data(
    store_config: Dict,
    loader_config: Dict,
    period: Dict,
    pairs: List[str],
) -> str:
    """
    Download market data from Binance to DuckDB storage.
    
    Args:
        pairs: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        date_range: Dict with 'start' and 'end' datetime configs
        storage_path: Path to DuckDB file
        
    Returns:
        Status message
    """
    
    store_type = store_config.get('type', 'duckdb/spot')
    store_params = store_config.get('params', {})

    if 'db_path' not in store_params:
        store_params['db_path'] = 'data/01_raw/market.duckdb'
    store_params['db_path'] = Path(store_params['db_path'])
    store_params['db_path'].parent.mkdir(parents=True, exist_ok=True)

    store: sf.data.raw_store.RawDataStore = sf.default_registry.get(
        component_type=sf.SfComponentType.RAW_DATA_STORE,
        name=store_type)(**store_params)

    loader_type = sf.default_registry.get(
        component_type=sf.SfComponentType.RAW_DATA_LOADER,
        name=loader_config.get('type', 'binance/spot'))
    
    loader_params = loader_config.get('params', {})
    
    loader: sf.data.raw_loader.RawDataLoader = loader_type(store=store, **loader_params)

    start = datetime(**period['start'])
    end = datetime(**period['end'])

    asyncio.run(loader.download(
        pairs=loader_params.get('pairs', ['BTCUSDT']),
        start=start,
        end=end,
    ))
    
    mlflow.log_params({
        "data.pairs": "[" + ", ".join(loader_params.get('pairs', ['BTCUSDT'])) + "]",
        "data.start_date": start.isoformat(),
        "data.end_date": end.isoformat(),
        "data.num_pairs": len(loader_params.get('pairs', ['BTCUSDT'])),
    })
    
    return str(store_params["db_path"])


def load_raw_data_from_storage(
    store_config: Dict,
    period: Dict,
    pairs: List[str],
    store_path: str,
) -> sf.RawData:
    db_path = Path(store_path)

    start = datetime(**period["start"])
    end = datetime(**period["end"])

    raw_data = sf.data.RawDataFactory.from_duckdb_spot_store(
        spot_store_path=db_path,
        pairs=pairs,
        start=start,
        end=end,
        data_types=["spot"],
    )

    spot_df = raw_data.get("spot")
    mlflow.log_metrics(
        {
            "data.total_rows": spot_df.height,
            "data.unique_pairs": spot_df.select("pair").n_unique(),
            "data.date_span_days": (end - start).days,
        }
    )
    return raw_data