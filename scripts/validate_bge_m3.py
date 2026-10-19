"""Verify BGE-M3 installation, cached weights, and real Arabic embedding output."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


MODEL_ID = "BAAI/bge-m3"
ARABIC_TEST_SENTENCE = "تدعم اللغة العربية البحث الدلالي واسترجاع المعلومات بدقة."


def _hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _find_snapshot() -> Path | None:
    snapshots = _hub_cache() / "models--BAAI--bge-m3" / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.iterdir(), reverse=True):
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            continue
        weights = (
            list(snapshot.glob("*.safetensors"))
            + list(snapshot.glob("pytorch_model*.bin"))
            + list(snapshot.glob("*.safetensors.index.json"))
            + list(snapshot.glob("*.bin.index.json"))
        )
        if weights:
            return snapshot
    return None


def _check_installation() -> tuple[Path | None, list[str]]:
    missing = [
        package
        for package in ("FlagEmbedding", "torch", "huggingface_hub")
        if importlib.util.find_spec(package) is None
    ]
    for package in ("FlagEmbedding", "torch", "huggingface_hub"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{package}: {version}")
    snapshot = _find_snapshot()
    print(f"Hugging Face cache: {_hub_cache()}")
    print(f"{MODEL_ID} cached: {'yes' if snapshot else 'no'}")
    if missing:
        print("Missing Python packages: " + ", ".join(missing))
    return snapshot, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device; auto selects CUDA when available.",
    )
    args = parser.parse_args()

    snapshot, missing = _check_installation()
    if missing or snapshot is None:
        print("\nBGE-M3 validation: BLOCKED")
        if missing:
            print(
                f'Install the dependencies with: "{sys.executable}" -m pip install '
                '"FlagEmbedding==1.3.5" "torch==2.5.1"'
            )
        if snapshot is None:
            print(
                f'Download the model with: "{sys.executable}" -c "from huggingface_hub '
                "import snapshot_download; print(snapshot_download(repo_id='BAAI/bge-m3'))\""
            )
        return 2

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        torch = importlib.import_module("torch")

        from src.chunking import TextChunk
        from src.embeddings import BGE_M3_DENSE_DIMENSIONS, EmbeddingService

        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch")

        chunk_id = hashlib.sha256(ARABIC_TEST_SENTENCE.encode("utf-8")).hexdigest()
        chunk = TextChunk(
            chunk_id=chunk_id,
            document_id=hashlib.sha256(b"bge-m3-validation").hexdigest(),
            chunk_index=0,
            page_number=1,
            strategy="recursive",
            text=ARABIC_TEST_SENTENCE,
            source_text=ARABIC_TEST_SENTENCE,
            estimated_token_count=10,
            char_count=len(ARABIC_TEST_SENTENCE),
            source_start=0,
            source_end=len(ARABIC_TEST_SENTENCE),
            source_filename="validation-input.pdf",
            page_language="arabic",
            extraction_method="native",
            source_metadata={"validation": True},
        )
        batch = EmbeddingService(
            model_name=str(snapshot),
            device=device,
            batch_size=1,
        ).embed_chunks([chunk])
        embedding = batch.embeddings[0]
        norm = sum(value * value for value in embedding.vector) ** 0.5
        if embedding.dimensions != BGE_M3_DENSE_DIMENSIONS:
            raise RuntimeError(
                f"Expected {BGE_M3_DENSE_DIMENSIONS} dimensions, got {embedding.dimensions}"
            )
        if abs(norm - 1.0) > 1e-4:
            raise RuntimeError(f"Expected a unit-normalized vector, got norm {norm:.6f}")

        print("\nBGE-M3 validation: PASS")
        print(f"Device: {device}")
        print(f"Sentence: {ARABIC_TEST_SENTENCE}")
        print(f"Vector dimensions: {embedding.dimensions}")
        print(f"Vector norm: {norm:.6f}")
        print(f"Generation seconds: {batch.performance.generation_seconds:.4f}")
        print(f"Chunks per second: {batch.performance.chunks_per_second:.2f}")
        print(f"Float32 storage per vector: {batch.bytes_per_vector:,} bytes")
        print(f"Generated at (UTC): {datetime.now(UTC).isoformat()}")
        return 0
    except Exception as exc:
        print(f"\nBGE-M3 validation: FAIL ({type(exc).__name__}: {exc})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())