"""Shared, reproducible configuration for preprocessing deliverables."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path = ROOT / "mtsamples.csv"
    output_dir: Path = ROOT / "outputs" / "preprocessing"
    nltk_data_dir: Path = ROOT / ".nltk_data"
    # None: keep every class meeting the independent-text support threshold.
    top_k: int | None = None
    min_class_groups: int = 100
    min_holdout_groups: int = 20
    benchmark_policy: str = "specialty_single_label"
    specialty_scope_path: Path = ROOT / "configs" / "specialty_scope.json"
    # Conservative normalization is the default; legacy cleaning is an ablation.
    cleaning_profile: str = "conservative"
    random_state: int = 42
    min_df: int = 3
    max_features: int = 5000
    keep_numbers: bool = True
    # Only for the optional source_labels baseline. Specialty mode resolves scope.
    conflict_policy: str = "group"
    download_nltk: bool = False
