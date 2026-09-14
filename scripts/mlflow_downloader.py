
import argparse
import mlflow
from pathlib import Path
from omegaconf import OmegaConf

def download_artifact(run_id: str, tracking_uri: str = "http://localhost:5000"):
    """Download and parse hydra_config.yaml from MLflow run artifacts."""
    artifact_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, tracking_uri=tracking_uri
    )
    return artifact_dir


def main(run_id: str, tracking_uri: str, output_dir: str):
    artifact_dir = download_artifact(run_id, tracking_uri=tracking_uri)
    config_path = Path(artifact_dir) / "hydra_config.yaml"
    cfg = OmegaConf.load(config_path)
    print(cfg)
    run_name = cfg.run_name
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.save(cfg, output_path / "hydra_config.yaml")


if __name__ == "__main__":
    ## example : python scripts/mlflow_downloader.py <run_id> --tracking-uri http://localhost:5000
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="MLflow run ID")
    parser.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--output-dir", default="../configs_downloaded", help="Output directory for downloaded artifacts")
    args = parser.parse_args()
    
    run_id = args.run_id
    tracking_uri = args.tracking_uri
    output_dir = args.output_dir
    
    main(run_id, tracking_uri, output_dir)