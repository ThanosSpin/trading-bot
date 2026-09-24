import pandas as pd

from predictive_model.model_monitor import evaluate_predictions


def test_evaluate_predictions_accepts_fresh_log_without_outcomes(tmp_path, capsys):
    pd.DataFrame(
        [
            {
                "timestamp": "2026-09-24T14:00:00+00:00",
                "symbol": "AAPL",
                "mode": "daily",
                "predicted_prob": 0.62,
                "price": 100.0,
            }
        ]
    ).to_csv(tmp_path / "predictions_AAPL.csv", index=False)

    result = evaluate_predictions(
        symbol="AAPL",
        mode="daily",
        lookback_days=7,
        logs_dir=str(tmp_path),
    )

    assert result["sample_size"] == 0
    assert "[ERROR]" not in capsys.readouterr().out
