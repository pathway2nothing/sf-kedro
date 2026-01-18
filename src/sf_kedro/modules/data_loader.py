"""Data loading and storage operations."""

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import mlflow

import signalflow as sf


def download_market_data(
    pairs: List[str],
    date_range: Dict,
    storage_path: str,
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
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    spot_store = sf.data.raw_store.DuckDbSpotStore(db_path=storage_path)
    loader = sf.data.source.BinanceSpotLoader(store=spot_store)
    
    # Convert date_range to datetime objects
    start = datetime(**date_range['start'])
    end = datetime(**date_range['end'])
    
    # Download data
    asyncio.run(loader.download(
        pairs=pairs,
        start=start,
        end=end,
    ))
    
    # Log to MLflow
    mlflow.log_params({
        "data.pairs": ",".join(pairs),
        "data.start_date": start.isoformat(),
        "data.end_date": end.isoformat(),
        "data.num_pairs": len(pairs),
    })
    
    return f"Downloaded {len(pairs)} pairs to {storage_path}"


def load_raw_data_from_storage(
    storage_path: str,
    pairs: List[str],
    date_range: Dict,
    data_types: List[str] = None,
) -> sf.core.RawData:
    """
    Load RawData from DuckDB storage.
    
    Args:
        storage_path: Path to DuckDB file
        pairs: List of trading pairs to load
        date_range: Date range for filtering
        data_types: Types of data to load (default: ["spot"])
        
    Returns:
        RawData object
    """
    if data_types is None:
        data_types = ["spot"]
    
    start = datetime(**date_range['start'])
    end = datetime(**date_range['end'])
    
    raw_data = sf.data.RawDataFactory.from_duckdb_spot_store(
        spot_store_path=Path(storage_path),
        pairs=pairs,
        start=start,
        end=end,
        data_types=data_types,
    )
    
    # Log data statistics
    spot_df = raw_data.get("spot")
    
    mlflow.log_metrics({
        "data.total_rows": spot_df.height,
        "data.unique_pairs": spot_df.select("pair").n_unique(),
        "data.date_span_days": (end - start).days,
    })
    
    return raw_data