"""Validator creation and loading."""


import mlflow

import signalflow as sf


def create_sklearn_validator(
    train_data: dict,
    val_data: dict,
    model_config: dict,
) -> sf.validator.SklearnSignalValidator:
    """
    Create and train sklearn validator.

    Args:
        train_data: Training data dict
        val_data: Validation data dict
        model_config: Model configuration

    Returns:
        Trained SklearnSignalValidator
    """
    from signalflow.validator import SklearnSignalValidator

    train_df = train_data["full"]
    feature_cols = [
        col for col in train_df.columns if col not in ["timestamp", "pair", "label"]
    ]

    X_train = train_df.select(feature_cols)
    y_train = train_df.select("label")

    model_type = model_config.pop("model_type")
    model_params = model_config.get("model_params", {})
    validator = SklearnSignalValidator(
        model_type=model_type,
        model_params=model_params,
        ts_col=model_config.get("ts_col", "timestamp"),
        pair_col=model_config.get("pair_col", "pair"),
        train_params=model_config.get("train_params", {}),
        tune_enabled=model_config.get("tune_enabled", False),
        tune_params=model_config.get("tune_params", {}),
    )
    validator.fit(X_train, y_train)

    val_df = val_data["full"]
    X_val = val_df.select(feature_cols)
    y_val = val_df.select("label")

    from sklearn.metrics import accuracy_score

    y_pred = validator.model.predict(X_val.to_pandas())
    val_accuracy = accuracy_score(y_val.to_pandas(), y_pred)

    mlflow.log_params(
        {"model.type": model_type, **{f"model.{k}": v for k, v in model_params.items()}}
    )

    mlflow.log_metrics(
        {
            "val_accuracy": val_accuracy,
        }
    )

    # Log model
    mlflow.sklearn.log_model(
        validator.model,
        artifact_path="sklearn_validator",
        registered_model_name=f"signalflow_sklearn_{model_type}",
    )

    return validator


def create_nn_validator(
    train_data: dict,
    val_data: dict,
    model_config: dict,
    trainer_config: dict,
) -> object:
    """
    Create and train neural network validator.

    Args:
        train_data: Training data dict
        val_data: Validation data dict
        model_config: Model architecture config
        trainer_config: Training config

    Returns:
        Trained neural validator
    """
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger

    from signalflow.nn.validator import TemporalValidator

    # Create validator
    validator = TemporalValidator(**model_config)

    # MLflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=mlflow.get_experiment(
            mlflow.active_run().info.experiment_id
        ).name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=mlflow.active_run().info.run_id,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.2f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=trainer_config.get("early_stopping_patience", 5),
        mode="min",
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=trainer_config.get("max_epochs", 50),
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=trainer_config.get("accelerator", "auto"),
        devices=trainer_config.get("devices", 1),
    )

    # Prepare dataloaders (simplified - implement properly)
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def create_dataloader(data_dict, batch_size=32):
        df = data_dict["full"]
        feature_cols = [
            col for col in df.columns if col not in ["timestamp", "pair", "label"]
        ]

        X = torch.tensor(df.select(feature_cols).to_numpy(), dtype=torch.float32)
        y = torch.tensor(
            df.select("label").to_pandas()["label"].astype("category").cat.codes.values,
            dtype=torch.long,
        )

        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_dataloader(
        train_data, batch_size=trainer_config.get("batch_size", 32)
    )
    val_loader = create_dataloader(
        val_data, batch_size=trainer_config.get("batch_size", 32)
    )

    # Train
    trainer.fit(validator, train_loader, val_loader)

    # Log best model
    mlflow.pytorch.log_model(
        validator.model,
        artifact_path="nn_validator",
        registered_model_name=f"signalflow_nn_{model_config.get('encoder_type', 'gru')}",
    )

    return validator


def load_validator_from_registry(
    model_name: str,
    stage: str = "Production",
) -> object:
    """
    Load validator from MLflow model registry.

    Args:
        model_name: Registered model name
        stage: Model stage (Production, Staging, etc.)

    Returns:
        Loaded validator model
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Get latest version in stage
    model_versions = client.get_latest_versions(model_name, stages=[stage])

    if not model_versions:
        raise ValueError(f"No model found for {model_name} in stage {stage}")

    model_version = model_versions[0]

    # Load model
    model_uri = f"models:/{model_name}/{stage}"

    # Detect model type and load accordingly
    if "sklearn" in model_name:
        model = mlflow.sklearn.load_model(model_uri)
        # Wrap in SklearnSignalValidator
        from signalflow.validator import SklearnSignalValidator

        validator = SklearnSignalValidator(model_type="loaded", model_params={})
        validator.model = model
    else:
        model = mlflow.pytorch.load_model(model_uri)
        # Wrap in appropriate validator
        validator = model  # Adjust based on your implementation

    # Log provenance
    mlflow.log_params(
        {
            "model.name": model_name,
            "model.version": model_version.version,
            "model.stage": stage,
            "model.run_id": model_version.run_id,
        }
    )

    return validator
