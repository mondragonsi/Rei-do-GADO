"""
BovSmart — Fine-Tuning para Detecção Aérea de Bovinos
======================================================
Baixa datasets públicos e faz fine-tuning do YOLOv8m para
detecção de bovinos em vídeos de drone (vista aérea).

Datasets usados (todos CC BY 4.0, gratuitos):
  1. ICAERUS Zenodo (10245396) — ~1.100 imagens de drone, anotadas, DJI Mavic
  2. Roboflow Aerial Cows (RF100) — 1.723 imagens de UAV, via API key gratuita

Uso:
  python finetune.py                         # fine-tuning padrão
  python finetune.py --epochs 100            # mais épocas
  python finetune.py --skip-download         # usar dataset já baixado
  python finetune.py --roboflow-key SUA_KEY  # incluir dataset Roboflow
"""

import argparse
import os
import shutil
import zipfile
import urllib.request
from pathlib import Path
import yaml

# ─── Caminhos ────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "finetune_data"
MODELS_DIR   = BASE_DIR / "models"
OUTPUT_MODEL = MODELS_DIR / "cow_aerial_finetuned.pt"

# ─── Modelo base (ICAERUS v2, já treinado para drone) ────────────────────────
BASE_MODEL_URL = (
    "https://raw.githubusercontent.com/ICAERUS-EU/UC3_Livestock_Monitoring"
    "/main/models/cow_detection/cow_detection_v2/cow_weight_v2_12.pt"
)
BASE_MODEL_PATH = MODELS_DIR / "cow_aerial_v2.pt"

# ─── Dataset Zenodo ICAERUS (anotado, CC BY 4.0) ─────────────────────────────
ZENODO_ANNOTATED_URL = "https://zenodo.org/records/10245396/files/Drone_images_cows_annotations.zip"
ZENODO_RAW_URL       = "https://zenodo.org/records/8234156/files/Drone_images_cows.zip"


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, label: str = ""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  ✓ Já existe: {dest.name}")
        return

    print(f"  ⬇  Baixando {label or dest.name} ...")
    done = [0]

    def _progress(count, block_size, total_size):
        pct = min(100, count * block_size * 100 // (total_size or 1))
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r     [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\r  ✓ Salvo em {dest}")


def extract_zip(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  📦 Extraindo {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"  ✓ Extraído em {dest}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: Zenodo ICAERUS
# ─────────────────────────────────────────────────────────────────────────────

def setup_zenodo_dataset() -> Path:
    """
    Baixa e organiza o dataset ICAERUS do Zenodo (~1.100 imagens com anotações
    em formato YOLO, capturadas por DJI Mavic a 30/60/100m de altitude).
    """
    zip_dest = DATA_DIR / "zenodo_icaerus.zip"
    extract_dir = DATA_DIR / "zenodo_icaerus"

    download_file(ZENODO_ANNOTATED_URL, zip_dest, "ICAERUS Zenodo (anotado)")

    if not extract_dir.exists():
        extract_zip(zip_dest, extract_dir)

    return extract_dir


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: Roboflow Aerial Cows (RF100) — requer API key gratuita
# ─────────────────────────────────────────────────────────────────────────────

def setup_roboflow_dataset(api_key: str) -> Path | None:
    """
    Baixa o dataset 'Aerial Cows' do Roboflow (1.723 imagens de UAV, CC BY 4.0).
    Requer uma API key gratuita de https://app.roboflow.com
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  ⚠  Instalando roboflow ...")
        os.system(
            "python -m pip install roboflow -q"
        )
        from roboflow import Roboflow

    rf_dir = DATA_DIR / "roboflow_aerial_cows"
    if rf_dir.exists() and any(rf_dir.rglob("*.jpg")):
        print("  ✓ Dataset Roboflow já baixado.")
        return rf_dir

    print("  ⬇  Baixando Aerial Cows do Roboflow (1.723 imagens) ...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-100").project("aerial-cows")
    dataset = project.version(2).download("yolov8", location=str(rf_dir))
    print(f"  ✓ Dataset Roboflow salvo em {rf_dir}")
    return rf_dir


# ─────────────────────────────────────────────────────────────────────────────
# Merge datasets into one unified YOLO structure
# ─────────────────────────────────────────────────────────────────────────────

def merge_datasets(dataset_dirs: list[Path]) -> Path:
    """
    Une múltiplos datasets YOLO em uma estrutura única:
      merged/
        images/train/  images/val/
        labels/train/  labels/val/
        data.yaml
    """
    merged = DATA_DIR / "merged"
    for split in ["train", "val"]:
        (merged / "images" / split).mkdir(parents=True, exist_ok=True)
        (merged / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_count = {"train": 0, "val": 0}

    for ds_dir in dataset_dirs:
        for split in ["train", "val"]:
            img_src = None
            lbl_src = None

            # Look for images and labels dirs in various common structures
            for candidate in [
                ds_dir / split / "images",
                ds_dir / "images" / split,
                ds_dir / split,
            ]:
                if candidate.exists() and any(candidate.glob("*.jpg")):
                    img_src = candidate
                    break

            for candidate in [
                ds_dir / split / "labels",
                ds_dir / "labels" / split,
            ]:
                if candidate.exists() and any(candidate.glob("*.txt")):
                    lbl_src = candidate
                    break

            if img_src is None:
                continue

            for img_file in img_src.glob("*.jpg"):
                stem = f"{ds_dir.name}_{img_file.stem}"
                shutil.copy(img_file, merged / "images" / split / f"{stem}.jpg")
                if lbl_src:
                    lbl_file = lbl_src / f"{img_file.stem}.txt"
                    if lbl_file.exists():
                        shutil.copy(lbl_file, merged / "labels" / split / f"{stem}.txt")
                img_count[split] += 1

    print(f"  ✓ Merged: {img_count['train']} treino | {img_count['val']} validação")

    # Write data.yaml
    data_yaml = {
        "path": str(merged.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["cow"],
    }
    yaml_path = merged / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Download base model
# ─────────────────────────────────────────────────────────────────────────────

def ensure_base_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_file(BASE_MODEL_URL, BASE_MODEL_PATH, "Modelo base ICAERUS v2 (93% precisão)")
    return BASE_MODEL_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(data_yaml: Path, base_model: Path, epochs: int, batch: int, imgsz: int):
    """
    Fine-tuning do YOLOv8m a partir do modelo ICAERUS (já treinado para drone).
    Usar como ponto de partida reduz drasticamente o número de épocas necessárias.
    """
    import torch
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  ⚠  GPU não detectada — treinamento na CPU (mais lento)")
        print("     Recomendado: Google Colab (GPU gratuita) para datasets grandes")
    else:
        print(f"  ✓ GPU detectada: {torch.cuda.get_device_name(0)}")

    print(f"\n  🚀 Iniciando fine-tuning por {epochs} épocas ...")
    print(f"     Base: {base_model.name}")
    print(f"     Dataset: {data_yaml}")
    print(f"     Dispositivo: {device}")
    print()

    model = YOLO(str(base_model))

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(MODELS_DIR / "runs"),
        name="cow_aerial_finetune",
        # Augmentações para imagens aéreas
        degrees=15.0,      # rotação (drone pode girar)
        flipud=0.5,        # flip vertical (vista aérea é rotação-invariante)
        fliplr=0.5,
        mosaic=1.0,        # mosaic augmentation
        hsv_h=0.015,       # variação de cor (luz do sol)
        hsv_s=0.4,
        hsv_v=0.4,
        # Regularização
        warmup_epochs=3,
        close_mosaic=10,
        patience=20,       # early stopping
        save_period=10,
        plots=True,
    )

    # Copy best.pt to models dir
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, OUTPUT_MODEL)
        print(f"\n  ✅ Modelo salvo em: {OUTPUT_MODEL}")
        print(f"     Métricas finais:")
        print(f"       mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"       Precisão: {results.results_dict.get('metrics/precision(B)', 'N/A'):.3f}")
        print(f"       Recall:   {results.results_dict.get('metrics/recall(B)', 'N/A'):.3f}")
    else:
        print("  ⚠  best.pt não encontrado. Verifique a pasta runs/")

    return OUTPUT_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BovSmart Fine-Tuning — Detecção Aérea de Bovinos"
    )
    parser.add_argument("--epochs",         type=int,   default=50,
                        help="Número de épocas (padrão: 50)")
    parser.add_argument("--batch",          type=int,   default=8,
                        help="Tamanho do batch (padrão: 8; reduza para 4 se der OOM)")
    parser.add_argument("--imgsz",          type=int,   default=640,
                        help="Tamanho da imagem (padrão: 640)")
    parser.add_argument("--roboflow-key",   type=str,   default=None,
                        help="API key do Roboflow (opcional). Crie grátis em roboflow.com")
    parser.add_argument("--skip-download",  action="store_true",
                        help="Pula download (usa dataset já existente em finetune_data/)")
    args = parser.parse_args()

    print("=" * 60)
    print("  🐄 BovSmart — Fine-Tuning Aéreo")
    print("=" * 60)

    # 1. Base model
    print("\n[1/4] Modelo base ICAERUS")
    base_model = ensure_base_model()

    # 2. Datasets
    dataset_dirs = []

    if not args.skip_download:
        print("\n[2/4] Baixando datasets")
        zenodo_dir = setup_zenodo_dataset()
        dataset_dirs.append(zenodo_dir)

        if args.roboflow_key:
            print()
            rf_dir = setup_roboflow_dataset(args.roboflow_key)
            if rf_dir:
                dataset_dirs.append(rf_dir)
        else:
            print("\n  ℹ  Para incluir o dataset Roboflow Aerial Cows (1.723 imgs extras):")
            print("     1. Crie conta gratuita em https://app.roboflow.com")
            print("     2. Vá em Settings → Roboflow API → copie sua Private API Key")
            print("     3. Rode: python finetune.py --roboflow-key SUA_KEY")
    else:
        # Use existing data
        zenodo_dir = DATA_DIR / "zenodo_icaerus"
        if zenodo_dir.exists():
            dataset_dirs.append(zenodo_dir)
        rf_dir = DATA_DIR / "roboflow_aerial_cows"
        if rf_dir.exists():
            dataset_dirs.append(rf_dir)
        print(f"\n[2/4] Usando datasets existentes: {[d.name for d in dataset_dirs]}")

    if not dataset_dirs:
        print("\n  ❌ Nenhum dataset encontrado. Rode sem --skip-download.")
        return

    # 3. Merge datasets
    print("\n[3/4] Organizando datasets")
    merged_dir = merge_datasets(dataset_dirs)
    data_yaml = merged_dir / "data.yaml"

    # 4. Fine-tune
    print("\n[4/4] Fine-tuning")
    run_finetune(
        data_yaml=data_yaml,
        base_model=base_model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
    )

    print("\n" + "=" * 60)
    print("  ✅ Fine-tuning concluído!")
    print(f"  📁 Modelo: {OUTPUT_MODEL}")
    print()
    print("  Para usar no BovSmart:")
    print("  1. Na sidebar, marque 'Usar modelo customizado (.pt)'")
    print(f"  2. Cole o caminho: {OUTPUT_MODEL}")
    print("  3. Defina cow_class_id = 0")
    print("=" * 60)


if __name__ == "__main__":
    main()
