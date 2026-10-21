from predictive_model.model_xgb import load_model

artifact = load_model("ABBV", "intraday_mom")

if artifact is None:
    raise RuntimeError("Artifact not found")

schema = artifact.get("feature_schema", {})
full_schema = artifact.get("full_feature_schema", {})

model = artifact["model"]

print("Model input width:", getattr(model, "n_features_in_", "unknown"))
print("Selected features:", len(artifact.get("features", [])))
print("Selected schema count:", schema.get("feature_count") if isinstance(schema, dict) else len(schema))
print("Full schema count:", full_schema.get("feature_count") if isinstance(full_schema, dict) else len(full_schema))