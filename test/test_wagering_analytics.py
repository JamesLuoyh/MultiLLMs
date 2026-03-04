"""
Unit tests for wagering analytics functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import Mock, MagicMock
import json

from wagering.training.analytics import WageringAnalytics
from wagering.methods import EqualWagers
from wagering.aggregation import LinearPooling


class MockWageringMethod:
    """Mock wagering method for testing."""
    
    def __init__(self, hidden_dim=64, hidden_layers=2, learning_rate=0.001, device_str="cpu"):
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.device_str = device_str


class MockAggregationFunction:
    """Mock aggregation function for testing."""
    
    pass


class TestWageringAnalytics:
    """Test WageringAnalytics class."""
    
    def test_create_training_analytics_basic(self):
        """Test creating basic training analytics dataframe."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        models = [Mock(), Mock()]
        datasets = [Mock(), Mock()]
        
        results = {
            "final_accuracy": 0.85,
            "final_nll": 0.5,
            "final_ece": 0.1,
            "final_auc": 0.9,
        }
        
        metadata = {
            "model_names": ["model1", "model2"],
            "dataset_names": ["dataset1", "dataset2"],
        }
        
        df = WageringAnalytics.create_training_analytics(
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            models=models,
            datasets=datasets,
            shuffle_data=True,
            shuffle_seed=42,
            early_stopping_patience=10,
            save_every=100,
            results=results,
            metadata=metadata,
        )
        
        # Check that it's a DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        
        # Check settings columns
        assert "wagering_method" in df.columns
        assert "aggregation_method" in df.columns
        assert "num_models" in df.columns
        assert "num_datasets" in df.columns
        assert "models" in df.columns
        assert "training_datasets" in df.columns
        assert "settings_hash" in df.columns
        assert "run_timestamp" in df.columns
        
        # Check result columns
        assert "final_accuracy" in df.columns
        assert "final_nll" in df.columns
        assert "final_ece" in df.columns
        assert "final_auc" in df.columns
        
        # Check values
        assert df["num_models"].iloc[0] == 2
        assert df["num_datasets"].iloc[0] == 2
        assert df["final_accuracy"].iloc[0] == 0.85
        assert df["result_type"].iloc[0] == "training"
    
    def test_create_training_analytics_with_checkpoint(self):
        """Test creating training analytics with checkpoint path."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            
            df = WageringAnalytics.create_training_analytics(
                wagering_method=wagering_method,
                aggregation_function=aggregation_function,
                models=[Mock()],
                datasets=[Mock()],
                shuffle_data=True,
                shuffle_seed=42,
                early_stopping_patience=10,
                save_every=100,
                results={"final_accuracy": 0.8},
                checkpoint_dir=checkpoint_dir,
            )
            
            assert "checkpoint_path" in df.columns
            assert str(checkpoint_dir) in df["checkpoint_path"].iloc[0]
    
    def test_create_training_analytics_settings_hash_consistency(self):
        """Test that same settings produce same settings_hash."""
        wagering_method1 = MockWageringMethod(hidden_dim=64, learning_rate=0.001)
        wagering_method2 = MockWageringMethod(hidden_dim=64, learning_rate=0.001)
        aggregation_function = MockAggregationFunction()
        
        results = {"final_accuracy": 0.8}
        metadata = {"model_names": ["model1"], "dataset_names": ["dataset1"]}
        
        df1 = WageringAnalytics.create_training_analytics(
            wagering_method=wagering_method1,
            aggregation_function=aggregation_function,
            models=[Mock()],
            datasets=[Mock()],
            shuffle_data=True,
            shuffle_seed=42,
            early_stopping_patience=10,
            save_every=100,
            results=results,
            metadata=metadata,
        )
        
        df2 = WageringAnalytics.create_training_analytics(
            wagering_method=wagering_method2,
            aggregation_function=aggregation_function,
            models=[Mock()],
            datasets=[Mock()],
            shuffle_data=True,
            shuffle_seed=42,
            early_stopping_patience=10,
            save_every=100,
            results={"final_accuracy": 0.9},  # Different result, same settings
            metadata=metadata,
        )
        
        # Same settings should produce same hash
        assert df1["settings_hash"].iloc[0] == df2["settings_hash"].iloc[0]
    
    def test_create_training_analytics_settings_hash_different(self):
        """Test that different settings produce different settings_hash."""
        wagering_method1 = MockWageringMethod(hidden_dim=64)
        wagering_method2 = MockWageringMethod(hidden_dim=128)  # Different
        aggregation_function = MockAggregationFunction()
        
        results = {"final_accuracy": 0.8}
        metadata = {"model_names": ["model1"], "dataset_names": ["dataset1"]}
        
        df1 = WageringAnalytics.create_training_analytics(
            wagering_method=wagering_method1,
            aggregation_function=aggregation_function,
            models=[Mock()],
            datasets=[Mock()],
            shuffle_data=True,
            shuffle_seed=42,
            early_stopping_patience=10,
            save_every=100,
            results=results,
            metadata=metadata,
        )
        
        df2 = WageringAnalytics.create_training_analytics(
            wagering_method=wagering_method2,
            aggregation_function=aggregation_function,
            models=[Mock()],
            datasets=[Mock()],
            shuffle_data=True,
            shuffle_seed=42,
            early_stopping_patience=10,
            save_every=100,
            results=results,
            metadata=metadata,
        )
        
        # Different settings should produce different hash
        assert df1["settings_hash"].iloc[0] != df2["settings_hash"].iloc[0]
    
    def test_create_evaluation_analytics_basic(self):
        """Test creating basic evaluation analytics dataframe."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        models = [Mock(), Mock()]
        
        results = {
            "accuracy": 0.85,
            "nll": 0.5,
            "auc": 0.9,
            "ece": 0.1,
            "num_examples": 100,
        }
        
        metadata = {"model_names": ["model1", "model2"]}
        
        df = WageringAnalytics.create_evaluation_analytics(
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            models=models,
            evaluation_dataset_name="test_dataset",
            training_datasets=["train_dataset1"],
            results=results,
            metadata=metadata,
            seed=42,
        )
        
        # Check that it's a DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        
        # Check settings columns
        assert "evaluation_dataset" in df.columns
        assert "training_datasets" in df.columns
        assert "models" in df.columns
        assert "seed" in df.columns
        assert "settings_hash" in df.columns
        assert "result_type" in df.columns
        
        # Check result columns
        assert "accuracy" in df.columns
        assert "nll" in df.columns
        assert "auc" in df.columns
        assert "ece" in df.columns
        
        # Check values
        assert df["evaluation_dataset"].iloc[0] == "test_dataset"
        assert df["accuracy"].iloc[0] == 0.85
        assert df["result_type"].iloc[0] == "evaluation"
    
    def test_create_evaluation_analytics_with_checkpoint(self):
        """Test creating evaluation analytics with checkpoint path."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        
        df = WageringAnalytics.create_evaluation_analytics(
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            models=[Mock()],
            evaluation_dataset_name="test_dataset",
            results={"accuracy": 0.8},
            checkpoint_path="/path/to/checkpoint",
        )
        
        assert "checkpoint_path" in df.columns
        assert df["checkpoint_path"].iloc[0] == "/path/to/checkpoint"
    
    def test_aggregate_results_by_settings_basic(self):
        """Test aggregating results from multiple runs with same settings."""
        # Create multiple dataframes with same settings but different results
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        
        dfs = []
        for i in range(5):
            df = WageringAnalytics.create_training_analytics(
                wagering_method=wagering_method,
                aggregation_function=aggregation_function,
                models=[Mock()],
                datasets=[Mock()],
                shuffle_data=True,
                shuffle_seed=42,
                early_stopping_patience=10,
                save_every=100,
                results={"final_accuracy": 0.7 + i * 0.05},  # Different results
                metadata={"model_names": ["model1"], "dataset_names": ["dataset1"]},
            )
            dfs.append(df)
        
        # Aggregate
        aggregated = WageringAnalytics.aggregate_results_by_settings(dfs)
        
        # Should have one row (same settings_hash)
        assert len(aggregated) == 1
        
        # Should have mean and std columns
        assert "final_accuracy" in aggregated.columns
        assert "final_accuracy_std" in aggregated.columns
        assert "num_runs" in aggregated.columns
        
        # Check that num_runs is correct
        assert aggregated["num_runs"].iloc[0] == 5
        
        # Check that mean is approximately correct
        expected_mean = np.mean([0.7 + i * 0.05 for i in range(5)])
        assert np.isclose(aggregated["final_accuracy"].iloc[0], expected_mean, atol=1e-6)
    
    def test_aggregate_results_by_settings_different_settings(self):
        """Test aggregating results from runs with different settings."""
        aggregation_function = MockAggregationFunction()
        
        # Create runs with different hidden_dim
        dfs = []
        for hidden_dim in [64, 128]:
            wagering_method = MockWageringMethod(hidden_dim=hidden_dim)
            df = WageringAnalytics.create_training_analytics(
                wagering_method=wagering_method,
                aggregation_function=aggregation_function,
                models=[Mock()],
                datasets=[Mock()],
                shuffle_data=True,
                shuffle_seed=42,
                early_stopping_patience=10,
                save_every=100,
                results={"final_accuracy": 0.8},
                metadata={"model_names": ["model1"], "dataset_names": ["dataset1"]},
            )
            dfs.append(df)
        
        # Aggregate
        aggregated = WageringAnalytics.aggregate_results_by_settings(dfs)
        
        # Should have two rows (different settings_hash)
        assert len(aggregated) == 2
        
        # Each should have num_runs = 1
        assert all(aggregated["num_runs"] == 1)
    
    def test_aggregate_results_by_settings_empty_list(self):
        """Test aggregating empty list returns empty DataFrame."""
        aggregated = WageringAnalytics.aggregate_results_by_settings([])
        assert isinstance(aggregated, pd.DataFrame)
        assert len(aggregated) == 0
    
    def test_aggregate_results_by_settings_custom_agg_functions(self):
        """Test aggregating with custom aggregation functions."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        
        dfs = []
        for i in range(3):
            df = WageringAnalytics.create_training_analytics(
                wagering_method=wagering_method,
                aggregation_function=aggregation_function,
                models=[Mock()],
                datasets=[Mock()],
                shuffle_data=True,
                shuffle_seed=42,
                early_stopping_patience=10,
                save_every=100,
                results={"final_accuracy": float(i)},
                metadata={"model_names": ["model1"], "dataset_names": ["dataset1"]},
            )
            dfs.append(df)
        
        # Use custom aggregation: min and max instead of mean/std
        agg_functions = {
            "final_accuracy": "min",
        }
        
        aggregated = WageringAnalytics.aggregate_results_by_settings(
            dfs, agg_functions=agg_functions
        )
        
        # Should have min value (column renamed to final_accuracy_min when using custom agg)
        assert aggregated["final_accuracy_min"].iloc[0] == 0.0
    
    def test_get_settings_columns(self):
        """Test getting settings columns from analytics dataframe."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        
        df = WageringAnalytics.create_training_analytics(
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            models=[Mock()],
            datasets=[Mock()],
            shuffle_data=True,
            shuffle_seed=42,
            early_stopping_patience=10,
            save_every=100,
            results={"final_accuracy": 0.8},
        )
        
        settings_cols = WageringAnalytics.get_settings_columns(df)
        
        # Should include settings columns but not result columns
        assert "wagering_method" in settings_cols
        assert "aggregation_method" in settings_cols
        assert "settings_hash" in settings_cols
        assert "final_accuracy" not in settings_cols
    
    def test_get_result_columns(self):
        """Test getting result columns from analytics dataframe."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        
        df = WageringAnalytics.create_training_analytics(
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            models=[Mock()],
            datasets=[Mock()],
            shuffle_data=True,
            shuffle_seed=42,
            early_stopping_patience=10,
            save_every=100,
            results={"final_accuracy": 0.8, "final_nll": 0.5},
        )
        
        result_cols = WageringAnalytics.get_result_columns(df)
        
        # Should include result columns
        assert "final_accuracy" in result_cols
        assert "final_nll" in result_cols
        assert "wagering_method" not in result_cols
    
    def test_load_and_aggregate_analytics(self):
        """Test loading and aggregating analytics from CSV files."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple CSV files
            csv_paths = []
            for i in range(3):
                df = WageringAnalytics.create_training_analytics(
                    wagering_method=wagering_method,
                    aggregation_function=aggregation_function,
                    models=[Mock()],
                    datasets=[Mock()],
                    shuffle_data=True,
                    shuffle_seed=42,
                    early_stopping_patience=10,
                    save_every=100,
                    results={"final_accuracy": 0.7 + i * 0.1},
                    metadata={"model_names": ["model1"], "dataset_names": ["dataset1"]},
                )
                csv_path = Path(tmpdir) / f"analytics_{i}.csv"
                df.to_csv(csv_path, index=False)
                csv_paths.append(str(csv_path))
            
            # Load and aggregate
            aggregated = WageringAnalytics.load_and_aggregate_analytics(csv_paths)
            
            # Should have one row (same settings)
            assert len(aggregated) == 1
            assert aggregated["num_runs"].iloc[0] == 3
            assert "final_accuracy" in aggregated.columns
            assert "final_accuracy_std" in aggregated.columns
    
    def test_load_and_aggregate_analytics_nonexistent_file(self):
        """Test loading with nonexistent file handles gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create one valid file
            wagering_method = MockWageringMethod()
            aggregation_function = MockAggregationFunction()
            
            df = WageringAnalytics.create_training_analytics(
                wagering_method=wagering_method,
                aggregation_function=aggregation_function,
                models=[Mock()],
                datasets=[Mock()],
                shuffle_data=True,
                shuffle_seed=42,
                early_stopping_patience=10,
                save_every=100,
                results={"final_accuracy": 0.8},
            )
            valid_path = Path(tmpdir) / "valid.csv"
            df.to_csv(valid_path, index=False)
            
            # Try to load with nonexistent file
            paths = [str(valid_path), str(Path(tmpdir) / "nonexistent.csv")]
            
            # Should handle gracefully and only load valid file
            aggregated = WageringAnalytics.load_and_aggregate_analytics(paths)
            assert len(aggregated) == 1
    
    def test_create_evaluation_analytics_multiple_datasets(self):
        """Test creating evaluation analytics for multiple datasets."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        
        # Create analytics for different evaluation datasets
        df1 = WageringAnalytics.create_evaluation_analytics(
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            models=[Mock()],
            evaluation_dataset_name="dataset1",
            results={"accuracy": 0.8},
        )
        
        df2 = WageringAnalytics.create_evaluation_analytics(
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            models=[Mock()],
            evaluation_dataset_name="dataset2",
            results={"accuracy": 0.9},
        )
        
        # Should have different evaluation_dataset values
        assert df1["evaluation_dataset"].iloc[0] == "dataset1"
        assert df2["evaluation_dataset"].iloc[0] == "dataset2"
        
        # Should have different settings_hash (different evaluation dataset)
        assert df1["settings_hash"].iloc[0] != df2["settings_hash"].iloc[0]
    
    def test_aggregate_results_by_settings_preserves_settings(self):
        """Test that aggregation preserves all settings columns."""
        wagering_method = MockWageringMethod(hidden_dim=64, learning_rate=0.001)
        aggregation_function = MockAggregationFunction()
        
        dfs = []
        for i in range(3):
            df = WageringAnalytics.create_training_analytics(
                wagering_method=wagering_method,
                aggregation_function=aggregation_function,
                models=[Mock()],
                datasets=[Mock()],
                shuffle_data=True,
                shuffle_seed=42,
                early_stopping_patience=10,
                save_every=100,
                results={"final_accuracy": 0.8},
                metadata={"model_names": ["model1"], "dataset_names": ["dataset1"]},
            )
            dfs.append(df)
        
        aggregated = WageringAnalytics.aggregate_results_by_settings(dfs)
        
        # Settings columns should be preserved (with 'first' suffix removed)
        assert "wagering_method" in aggregated.columns
        assert "aggregation_method" in aggregated.columns
        assert "wagering_hidden_dim" in aggregated.columns
        assert "wagering_learning_rate" in aggregated.columns
        assert "models" in aggregated.columns
        
        # Check that settings values are correct
        assert aggregated["wagering_hidden_dim"].iloc[0] == 64
        assert aggregated["wagering_learning_rate"].iloc[0] == 0.001
    
    def test_create_training_analytics_nan_handling(self):
        """Test that NaN values in results are handled correctly."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        
        # Results with NaN
        results = {
            "final_accuracy": 0.8,
            "final_nll": 0.5,
            "final_ece": np.nan,
            "final_auc": np.nan,
        }
        
        df = WageringAnalytics.create_training_analytics(
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            models=[Mock()],
            datasets=[Mock()],
            shuffle_data=True,
            shuffle_seed=42,
            early_stopping_patience=10,
            save_every=100,
            results=results,
        )
        
        # NaN values should be converted to None
        assert pd.isna(df["final_ece"].iloc[0]) or df["final_ece"].iloc[0] is None
        assert pd.isna(df["final_auc"].iloc[0]) or df["final_auc"].iloc[0] is None
    
    def test_create_evaluation_analytics_nan_handling(self):
        """Test that NaN values in evaluation results are handled correctly."""
        wagering_method = MockWageringMethod()
        aggregation_function = MockAggregationFunction()
        
        results = {
            "accuracy": 0.8,
            "nll": 0.5,
            "auc": np.nan,
            "ece": np.nan,
        }
        
        df = WageringAnalytics.create_evaluation_analytics(
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            models=[Mock()],
            evaluation_dataset_name="test",
            results=results,
        )
        
        # NaN values should be converted to None
        assert pd.isna(df["auc"].iloc[0]) or df["auc"].iloc[0] is None
        assert pd.isna(df["ece"].iloc[0]) or df["ece"].iloc[0] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

