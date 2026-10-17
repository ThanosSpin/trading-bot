# Unified intraday research protocol

This branch is research-only. It does not promote artifacts or change paper/live execution.

## Fixed experiment

- Data: frozen Alpaca IEX regular-session 15-minute bars.
- Development: all sessions before the final 63 trading sessions.
- Historical validation: the final 63 sessions, scored once after a design is locked.
- Models: scaled L2 logistic regression and strongly regularized XGBoost.
- Costs: round-trip basis points supplied on the command line.
- Targets:
  - 30 minutes, 0.50 volatility units.
  - 30 minutes, 0.75 volatility units.
  - 60 minutes, 0.50 volatility units.
  - 60 minutes, 0.75 volatility units.
- A positive target requires the future return to exceed the larger of transaction cost and the volatility-scaled move.
- Selection requires the existing fold stability gates. Aggregate return alone cannot select a design.

The historical validation period is not described as untouched because earlier experiments inspected folds drawn from the same year. Only future paper observations collected after the design lock are a genuinely untouched final evaluation.

## Workflow

1. Freeze the Parquet files and create their content hashes:

   ```bash
   python freeze_research_dataset.py \
     --result logs/unified_intraday/unified_intraday_BASELINE_NVDA.json \
     --result logs/unified_intraday/unified_intraday_BASELINE_AAPL.json
   ```

2. Run all predefined development designs without accessing the reserved sessions:

   ```bash
   python unified_intraday_research.py \
     --manifest research_manifests/intraday_15m_frozen.json \
     --target-design all \
     --cost-bps 10
   ```

3. Lock at most one stable design per symbol:

   ```bash
   python lock_unified_design.py logs/unified_intraday/unified_intraday_TIMESTAMP.json
   ```

   An empty lock is a valid research result: no design qualified.

4. If the lock contains selections, score it once on historical validation:

   ```bash
   python validate_locked_design.py research_locks/unified_intraday_lock.json
   ```

5. Do not retune from historical-validation results. A surviving design must next run unchanged in paper mode on observations produced after the lock time.
