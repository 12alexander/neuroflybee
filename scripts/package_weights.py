"""Empaqueta y verifica los pesos entrenados (v19, C5: «pesos disponibles mediante un enlace»).

    python scripts/package_weights.py pack       # checkpoints/MANIFEST.json + neuroflybee_pesos.zip
    python scripts/package_weights.py verify     # tras descargar: comprueba tamaño y SHA-256
    python scripts/package_weights.py upload --repo usuario/neuroflybee   # Hugging Face (opcional)

El manifiesto (con la versión del código y el hash de cada archivo) se publica junto al
.zip: quien descargue los pesos puede comprobar que son exactamente los del artículo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee import __version__  # noqa: E402

PATTERNS = ("*.pt", "*.json")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(checkpoints: Path) -> dict:
    files = sorted({p for pat in PATTERNS for p in checkpoints.glob(pat) if p.name != "MANIFEST.json"})
    return {"proyecto": "NeuroFly-Bee Perú", "version_codigo": __version__,
            "generado": datetime.now().isoformat(timespec="seconds"),
            "archivos": [{"archivo": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]}


def pack(checkpoints: Path, out_zip: Path) -> dict:
    manifest = build_manifest(checkpoints)
    if not manifest["archivos"]:
        raise SystemExit(f"No hay pesos en {checkpoints}: entrene primero (scripts/run_pipeline.py)")
    (checkpoints / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in manifest["archivos"]:
            zf.write(checkpoints / entry["archivo"], f"checkpoints/{entry['archivo']}")
        zf.write(checkpoints / "MANIFEST.json", "checkpoints/MANIFEST.json")
    return manifest


def verify(checkpoints: Path) -> list[str]:
    """Devuelve la lista de problemas (vacía si todo coincide con el manifiesto)."""
    manifest_path = checkpoints / "MANIFEST.json"
    if not manifest_path.exists():
        return [f"falta {manifest_path}"]
    problems = []
    for entry in json.loads(manifest_path.read_text(encoding="utf-8"))["archivos"]:
        p = checkpoints / entry["archivo"]
        if not p.exists():
            problems.append(f"falta {entry['archivo']}")
        elif p.stat().st_size != entry["bytes"] or sha256(p) != entry["sha256"]:
            problems.append(f"{entry['archivo']} no coincide con el manifiesto")
    return problems


def upload(checkpoints: Path, out_zip: Path, repo: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install huggingface_hub  (y `huggingface-cli login` con un token de escritura)")
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(checkpoints), repo_id=repo, repo_type="model", path_in_repo="checkpoints")
    print(f"Pesos publicados en https://huggingface.co/{repo}  → poner el enlace en README y paper")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("accion", choices=["pack", "verify", "upload"])
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--zip", default=str(ROOT / "neuroflybee_pesos.zip"))
    parser.add_argument("--repo", help="repositorio de Hugging Face, p. ej. usuario/neuroflybee")
    args = parser.parse_args(argv)
    ck = Path(args.checkpoints)
    if args.accion == "pack":
        m = pack(ck, Path(args.zip))
        size = sum(e["bytes"] for e in m["archivos"]) / 1e6
        print(f"{len(m['archivos'])} archivos ({size:.1f} MB) → {args.zip} y {ck / 'MANIFEST.json'}")
    elif args.accion == "verify":
        problems = verify(ck)
        print("Pesos verificados: coinciden con el manifiesto" if not problems else "\n".join(problems))
        raise SystemExit(1 if problems else 0)
    else:
        if not args.repo:
            raise SystemExit("--repo usuario/nombre es obligatorio para upload")
        upload(ck, Path(args.zip), args.repo)


if __name__ == "__main__":
    main()
