# sf_kedro/custom_modules/classification_metrics.py

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
)

import signalflow as sf
from .signal_metrics import SignalMetricsProcessor

@dataclass
@sf.sf_component(name="classification")
class SignalClassificationMetrics(SignalMetricsProcessor):
    """Analyze signal classification performance against labels.
    
    Computes standard classification metrics including:
    - Precision, Recall, F1 Score
    - Confusion Matrix
    - ROC Curve and AUC
    - Signal strength distribution
    
    Requires labels to be provided.
    """
    
    # Label mapping configuration
    positive_labels: list = None  # e.g., ['rise', 'up', 1]
    negative_labels: list = None  # e.g., ['fall', 'down', 0]
    
    chart_height: int = 900
    chart_width: int = 1400
    roc_n_thresholds: int = 100
    
    def __post_init__(self):
        """Set default label mappings if not provided."""
        if self.positive_labels is None:
            self.positive_labels = ['rise', 'up', 1, 'positive', 'buy']
        if self.negative_labels is None:
            self.negative_labels = ['fall', 'down', 0, 'negative', 'sell']
    
    def _map_labels_to_binary(self, labels: np.ndarray) -> np.ndarray:
        """Convert string/mixed labels to binary (0/1).
        
        Args:
            labels: Array of labels (can be strings, ints, etc.)
            
        Returns:
            Binary numpy array (0 for negative, 1 for positive)
        """
        binary_labels = np.zeros(len(labels), dtype=int)
        
        for i, label in enumerate(labels):
            if label in self.positive_labels:
                binary_labels[i] = 1
            elif label in self.negative_labels:
                binary_labels[i] = 0
            else:
                logger.warning(f"Unknown label value: {label}, treating as negative")
                binary_labels[i] = 0
        
        return binary_labels
    
    def compute(
        self,
        raw_data: sf.RawData,
        signals: sf.Signals,
        labels: pl.DataFrame | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute classification metrics."""
        
        if labels is None:
            logger.error("Labels are required for classification metrics")
            return None, {}
        
        signals_df = signals.value
        
        # Join signals with labels
        signals_with_labels = signals_df.join(
            labels,
            on=["timestamp", "pair"],
            how="inner"
        )
        
        # Filter only non-zero signals (actual predictions)
        predictions = signals_with_labels.filter(
            pl.col("signal") != 0
        )
        
        if predictions.height == 0:
            logger.warning("No non-zero signals found for classification")
            return None, {}
        
        logger.info(f"Found {predictions.height} signal-label pairs for classification")
        
        # Extract predictions and true labels
        y_pred = predictions["signal"].to_numpy()
        y_true_raw = predictions["label"].to_numpy()
        
        # Log unique label values for debugging
        unique_labels = np.unique(y_true_raw)
        logger.info(f"Unique label values: {unique_labels}")
        logger.info(f"Unique prediction values: {np.unique(y_pred)}")
        
        # Convert labels to binary
        y_true = self._map_labels_to_binary(y_true_raw)
        
        # Convert predictions to binary (1 for positive signal, 0 for negative)
        # Assuming signal values: 1 (buy), -1 (sell)
        y_pred_binary = (y_pred > 0).astype(int)
        
        logger.info(f"After conversion - Unique y_true: {np.unique(y_true)}, y_pred: {np.unique(y_pred_binary)}")
        
        # Get signal strengths if available
        if "strength" in predictions.columns:
            strengths = predictions["strength"].to_numpy()
        else:
            # Use absolute signal value as strength
            strengths = np.abs(y_pred)
        
        # Compute confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                logger.warning(f"Unexpected confusion matrix shape: {cm.shape}")
                tn, fp, fn, tp = 1, 1, 1, 1
        except Exception as e:
            logger.warning(f"Could not compute confusion matrix: {e}, using defaults")
            tn, fp, fn, tp = 1, 1, 1, 1
        
        # Basic metrics
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Specificity and sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2
        
        # Positive rate
        positive_rate = np.mean(y_true)
        
        # ROC Curve
        if len(np.unique(strengths)) > 1 and len(np.unique(y_true)) > 1:
            thresholds = np.linspace(
                strengths.min(),
                strengths.max(),
                self.roc_n_thresholds
            )
            tpr_list, fpr_list = [], []
            
            for threshold in thresholds:
                threshold_preds = (strengths >= threshold).astype(int)
                try:
                    cm_t = confusion_matrix(y_true, threshold_preds)
                    if cm_t.shape == (2, 2):
                        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
                    else:
                        continue
                except Exception:
                    continue
                    
                tpr = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
                fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
                tpr_list.append(tpr)
                fpr_list.append(fpr)
            
            # Compute AUC
            if len(tpr_list) > 1:
                auc = np.trapz(tpr_list[::-1], fpr_list[::-1])
            else:
                auc = 0.5
        else:
            thresholds = np.array([strengths.min()])
            tpr_list, fpr_list = [sensitivity], [1 - specificity]
            auc = 0.5
            logger.warning("Not enough unique values for ROC curve, using defaults")
        
        # Log loss
        logloss = np.nan
        try:
            if len(np.unique(y_true)) > 1 and len(np.unique(strengths)) > 1:
                # Normalize strengths to probabilities [0, 1]
                strength_range = strengths.max() - strengths.min()
                if strength_range > 0:
                    probs = (strengths - strengths.min()) / strength_range
                    # Clip to avoid log(0)
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    logloss = log_loss(y_true=y_true, y_pred=probs, labels=[0, 1])
        except Exception as e:
            logger.warning(f"Could not compute log loss: {e}")
        
        # Strength statistics
        strength_mean = float(np.mean(strengths))
        strength_std = float(np.std(strengths)) if len(strengths) > 1 else 0.0
        strength_quartiles = np.percentile(strengths, [25, 50, 75]).tolist()
        
        computed_metrics = {
            "quant": {
                "total_signals": int(predictions.height),
                "total_positive_signals": int(tp + fp),
                "total_negative_signals": int(tn + fn),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "sensitivity": float(sensitivity),
                "balanced_accuracy": float(balanced_acc),
                "f1": float(f1),
                "positive_rate": float(positive_rate),
                "auc": float(auc),
                "log_loss": float(logloss) if not np.isnan(logloss) else None,
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                },
                "strength_mean": strength_mean,
                "strength_std": strength_std,
            },
            "series": {
                "roc_curve": {
                    "tpr": tpr_list,
                    "fpr": fpr_list,
                    "thresholds": thresholds.tolist(),
                },
                "strength_quartiles": strength_quartiles,
            },
        }
        
        plots_context = {
            "total_samples": predictions.height,
            "label_mapping": {
                "positive": self.positive_labels,
                "negative": self.negative_labels,
            }
        }
        
        logger.info(
            f"Classification metrics computed: "
            f"Precision={precision:.3f}, Recall={recall:.3f}, "
            f"F1={f1:.3f}, AUC={auc:.3f}"
        )
        
        return computed_metrics, plots_context
    
    def plot(
        self,
        computed_metrics: Dict[str, Any],
        plots_context: Dict[str, Any],
        raw_data: sf.RawData,
        signals: sf.Signals,
        labels: pl.DataFrame | None = None,
    ) -> go.Figure:
        """Generate classification metrics visualization."""
        
        if computed_metrics is None:
            logger.error("No metrics available for plotting")
            return None
        
        fig = self._create_figure()
        
        self._add_roc_curve(fig, computed_metrics)
        self._add_confusion_matrix(fig, computed_metrics)
        self._add_strength_distribution(fig, computed_metrics)
        self._add_metrics_table(fig, computed_metrics)
        self._update_layout(fig)
        
        return fig
    
    @staticmethod
    def _create_figure():
        """Create subplot structure."""
        return make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "ROC Curve",
                "Confusion Matrix",
                "Signal Strength Distribution",
                "Key Metrics",
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "table"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )
    
    @staticmethod
    def _add_roc_curve(fig, metrics):
        """Add ROC curve plot."""
        roc = metrics["series"]["roc_curve"]
        auc = metrics["quant"]["auc"]
        
        # ROC Curve
        fig.add_trace(
            go.Scatter(
                x=roc["fpr"],
                y=roc["tpr"],
                mode="lines",
                name="ROC Curve",
                line=dict(color="blue", width=2),
                hovertemplate=(
                    "<b>FPR:</b> %{x:.3f}<br>"
                    "<b>TPR:</b> %{y:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        
        # Random classifier line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="red", dash="dash", width=2),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        
        # AUC annotation
        fig.add_annotation(
            x=0.5,
            y=0.1,
            text=f"<b>AUC = {auc:.3f}</b>",
            showarrow=False,
            font=dict(size=14, color="blue"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="blue",
            borderwidth=1,
            borderpad=4,
            row=1,
            col=1,
        )
    
    @staticmethod
    def _add_confusion_matrix(fig, metrics):
        """Add confusion matrix heatmap."""
        cm = metrics["quant"]["confusion_matrix"]
        cm_values = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        total = sum(cm.values())
        cm_pcts = [[val / total * 100 for val in row] for row in cm_values]
        
        # Heatmap
        fig.add_trace(
            go.Heatmap(
                z=cm_values,
                x=["Predicted Negative", "Predicted Positive"],
                y=["Actual Negative", "Actual Positive"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(x=0.95),
                hovertemplate=(
                    "<b>%{y} / %{x}</b><br>"
                    "Count: %{z}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        
        # Add text annotations
        annotations = []
        for i in range(2):
            for j in range(2):
                annotations.append(
                    dict(
                        text=f"<b>{cm_values[i][j]}</b><br>({cm_pcts[i][j]:.1f}%)",
                        x=["Predicted Negative", "Predicted Positive"][j],
                        y=["Actual Negative", "Actual Positive"][i],
                        showarrow=False,
                        font=dict(
                            color="white" if cm_values[i][j] > total/4 else "black",
                            size=12
                        ),
                    )
                )
        
        for annotation in annotations:
            fig.add_annotation(annotation, row=1, col=2)
    
    @staticmethod
    def _add_strength_distribution(fig, metrics):
        """Add signal strength distribution plot."""
        mean = metrics["quant"]["strength_mean"]
        std = metrics["quant"]["strength_std"]
        quartiles = metrics["series"]["strength_quartiles"]
        
        # Handle case where std is 0
        if std < 1e-10:
            std = 0.01
        
        # Generate normal distribution curve
        x_values = np.linspace(mean - 3*std, mean + 3*std, 200)
        y_values = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std) ** 2)
        
        # Distribution curve
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name="Strength Distribution",
                fill="tozeroy",
                line=dict(color="green", width=2),
                fillcolor="rgba(0, 255, 0, 0.1)",
                hovertemplate=(
                    "<b>Strength:</b> %{x:.3f}<br>"
                    "<b>Density:</b> %{y:.4f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        
        # Quartile lines
        quartile_colors = ["rgba(255,0,0,0.6)", "rgba(0,255,0,0.6)", "rgba(0,0,255,0.6)"]
        quartile_names = ["Q1 (25%)", "Q2 (50%)", "Q3 (75%)"]
        
        for q_val, color, name in zip(quartiles, quartile_colors, quartile_names):
            fig.add_vline(
                x=q_val,
                line_color=color,
                line_dash="dash",
                line_width=2,
                annotation_text=name,
                annotation_position="top",
                row=2,
                col=1,
            )
    
    @staticmethod
    def _add_metrics_table(fig, metrics):
        """Add metrics summary table."""
        quant = metrics["quant"]
        
        table_data = [
            ["Total Signals", f"{quant['total_signals']}"],
            ["Positive Signals", f"{quant['total_positive_signals']}"],
            ["Precision", f"{quant['precision']:.3f}"],
            ["Recall", f"{quant['recall']:.3f}"],
            ["Specificity", f"{quant['specificity']:.3f}"],
            ["Sensitivity", f"{quant['sensitivity']:.3f}"],
            ["F1 Score", f"{quant['f1']:.3f}"],
            ["Balanced Accuracy", f"{quant['balanced_accuracy']:.3f}"],
            ["Positive Rate", f"{quant['positive_rate']:.3f}"],
            ["Mean Strength", f"{quant['strength_mean']:.3f}"],
            ["Strength Std", f"{quant['strength_std']:.3f}"],
        ]
        
        if quant['log_loss'] is not None:
            table_data.append(["Log Loss", f"{quant['log_loss']:.3f}"])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>"],
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=12, color="black"),
                    height=30,
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color="lavender",
                    align="left",
                    font=dict(size=11, color="black"),
                    height=25,
                ),
            ),
            row=2,
            col=2,
        )
    
    def _update_layout(self, fig):
        """Update figure layout and axes."""
        fig.update_xaxes(
            title_text="False Positive Rate",
            range=[0, 1],
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="True Positive Rate",
            range=[0, 1],
            row=1,
            col=1,
        )
        
        fig.update_xaxes(
            title_text="Predicted Class",
            row=1,
            col=2,
        )
        fig.update_yaxes(
            title_text="Actual Class",
            row=1,
            col=2,
        )
        
        fig.update_xaxes(
            title_text="Signal Strength",
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Probability Density",
            row=2,
            col=1,
        )
        
        fig.update_layout(
            title=dict(
                text="SignalFlow: Classification Performance Analysis<br>"
                     "<sub>ROC, Confusion Matrix, and Strength Distribution</sub>",
                font=dict(color="black", size=16),
                x=0.5,
                xanchor="center",
            ),
            height=self.chart_height,
            width=self.chart_width,
            template="plotly_white",
            showlegend=True,
        )