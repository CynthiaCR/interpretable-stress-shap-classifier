"""Download the Dreaddit stress classification dataset from Hugging Face."""

from pathlib import Path
from datasets import load_dataset

DATASET_ID = "asmaab/dreaddit"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def save_dreaddit(dataset, output_dir: Path = DATA_DIR) -> None:
    """Write each split to CSV under output_dir."""
    
    if output_dir.exists():
        print(f"Path already exists: {output_dir}. Skipping download.")
        return

    output_dir.mkdir(parents=True)
    for split, rows in dataset.items():
        out_path = output_dir / f"{split}.csv"
        rows.to_csv(out_path, index=False)
        print(f"{split}: wrote {len(rows)} rows -> {out_path}")

if __name__ == "__main__":
    dataset = load_dataset(DATASET_ID)
    save_dreaddit(dataset)
