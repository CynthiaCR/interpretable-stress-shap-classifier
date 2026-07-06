"""Download the Dreaddit stress classification dataset from Hugging Face."""

from pathlib import Path

from datasets import load_dataset

DATASET_ID = "asmaab/dreaddit"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def save_dreaddit(dataset, output_dir: Path = DATA_DIR) -> None:
    """Write each split to CSV under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in dataset.items():
        print(f"{split}: {len(rows)} rows")
        rows.to_csv(output_dir / f"{split}.csv")
        print(f"-> Output dir: {output_dir / f'{split}.csv'}\n")


if __name__ == "__main__":
    dataset = load_dataset(DATASET_ID)
    save_dreaddit(dataset)
