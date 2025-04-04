
import os
import json
import torch
import mlflow
from pathlib import Path
from model_pool import ConvNeXtRegressor
from huggingface_hub import create_repo, upload_folder

from torchvision.models import convnext_tiny  # change if you're using another variant

HF_REPO_ID = "Oucru-Vital/Respiratory-Rate-PPG-ECG"  # change if needed
RUN_ID = "20c8575629f64913bb262e100bef7167"  # set to specific MLflow run ID or leave None to use latest


# === 1. Locate latest MLflow run and checkpoint ===
client = mlflow.tracking.MlflowClient()
run = client.get_run(RUN_ID or client.search_runs("0", order_by=["start_time DESC"])[0].info.run_id)
run_id = run.info.run_id
print(f"Using MLflow run: {run_id}")

# === 2. Download raw .pt checkpoint ===
# Assume the artifact path is like: "model/vit_best.pt"
checkpoint_path = "best_model.pth"  # update if different
local_checkpoint = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=checkpoint_path)

# === 3. Load model manually ===
model = ConvNeXtRegressor()
model.load_state_dict(torch.load(local_checkpoint, map_location="cpu"))

# === 4. Export state_dict + metadata ===
output_dir = Path("hf_model")
output_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), output_dir / "convnext_state_dict.pt")

config = {
    "backbone": "convnext_tiny",
    "input_shape": [3, 224, 224],
    "output_dim": 1,
    "description": "ConvNeXt model trained to predict respiratory rate from PPG+ECG spectrograms."
}
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# === 5. Dump MLflow metrics and params ===
metrics = run.data.metrics
params = run.data.params

with open(output_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(output_dir / "params.json", "w") as f:
    json.dump(params, f, indent=2)

# === 6. Create README.md ===
readme = f"""# Respiratory Rate Estimation Model (ConvNeXt)

This model uses a `torchvision.models.convnext_tiny` backbone and was trained to predict respiratory rate from spectrogram images derived from PPG and ECG signals.

## How to Use

```python
import torch
from model import load_model

model = load_model("convnext_state_dict.pt")
model.eval()
```

## Metrics (from MLflow)

- Validation MAE: {metrics.get('val_mae', 'N/A')}
- Run ID: {run_id}

## GitHub

👉 https://github.com/Oucru-Innovations/Respiratory-Rate-PPG-ECG
"""
with open(output_dir / "README.md", "w") as f:
    f.write(readme)

# === 7. Upload to Hugging Face ===
create_repo(HF_REPO_ID, exist_ok=True, repo_type="model")
upload_folder(
    folder_path=output_dir,
    repo_id=HF_REPO_ID,
    repo_type="model"
)

print(f"\n✅ Model uploaded to https://huggingface.co/{HF_REPO_ID}")
