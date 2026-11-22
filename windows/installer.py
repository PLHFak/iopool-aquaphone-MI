# -*- coding: utf-8 -*-
# Aquaphone-MI : installation dans WSL2 (Ubuntu) sur le PC Windows équipé d'une RTX 4090.
# MI = surcouche locale (Mistral) du dépôt IOPOOL-aquaphone : on clone le code de base,
# puis on pose par-dessus les fichiers MI (labo/serveur.py, labo/web/index.html).
# Idempotent : la 2e fois, quelques secondes.  À lancer DANS WSL2 :
#   bash ~/iopool-aquaphone-MI/windows/installer.sh
import os
import shutil
import subprocess
import sys
from pathlib import Path

MAISON = Path.home()
K = MAISON / "iopool-aquaphone-MI"   # ce kit MI
D = MAISON / "iopool-aquaphone"      # code de base (dépôt PLHFak/IOPOOL-aquaphone)


def info(m): print(f"\n== {m}")
def erreur(m): print(f"ERREUR : {m}")


def run(cmd, **kw):
    return subprocess.run(cmd, **kw)


info("Aquaphone-MI : vérifications")
if run(["bash", "-lc", "command -v nvidia-smi"]).returncode == 0:
    run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
else:
    print("  ATTENTION : nvidia-smi introuvable dans WSL. Vérifiez que WSL2 est bien actif")
    print("  (pas WSL1) et que le pilote NVIDIA Windows est à jour.")

info("Code de base (aquaphone)")
if not D.exists():
    run(["git", "clone", "https://github.com/PLHFak/IOPOOL-aquaphone.git", str(D)])
run(["git", "-C", str(D), "pull", "--ff-only"])

info("Surcouche MI (labo/serveur.py + page) : HD 720, compression adaptative, menu explicite")
shutil.copy(K / "labo" / "serveur.py", D / "labo" / "serveur.py")
shutil.copy(K / "labo" / "web" / "index.html", D / "labo" / "web" / "index.html")
shutil.copy(K / "labo2" / "clips_local.py", D / "labo2" / "clips_local.py")
env = D / ".env"
lignes = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
if env.exists():
    if not any(l.startswith("AQUAPHONE_LARGEUR=") for l in lignes):
        lignes.append("AQUAPHONE_LARGEUR=720")
        env.write_text("\n".join(lignes) + "\n", encoding="utf-8")
else:
    print("  (pas encore de .env : à créer avec ANTHROPIC_API_KEY et AQUAPHONE_MOT_DE_PASSE)")

info("Python (venv dans le dépôt de base)")
if run(["bash", "-lc", "command -v python3"]).returncode != 0:
    run(["sudo", "apt-get", "update", "-qq"])
    run(["sudo", "apt-get", "install", "-y", "-qq", "python3-venv", "python3-pip"])
if not (D / ".venv").exists():
    run([sys.executable, "-m", "venv", str(D / ".venv")])
pip = str(D / ".venv" / "bin" / "pip")
run([pip, "install", "-q", "--upgrade", "pip"])
run([pip, "install", "-q", "-r", str(D / "requirements.txt")])
run(["sudo", "apt-get", "install", "-y", "-qq", "ffmpeg", "libegl1", "libgl1", "libgles2", "sox"])

info("Ditto (visage) — même version que sur le pod + correctif de recollage OpenCV")
if not (D / "third_party" / "ditto").exists():
    run(["git", "clone", "-q", "https://github.com/antgroup/ditto-talkinghead",
         str(D / "third_party" / "ditto")])
    run(["git", "-C", str(D / "third_party" / "ditto"), "checkout", "-q", "c3e47ee"])
    run(["git", "-C", str(D / "third_party" / "ditto"), "apply",
         str(D / "labo" / "patches" / "ditto_blend_numpy.patch")])

info("Modèles (≈ 8 Go au 1er lancement)")
os.environ["HF_HOME"] = str(MAISON / ".cache" / "huggingface")
hf = str(D / ".venv" / "bin" / "hf")
run([hf, "download", "digital-avatar/ditto-talkinghead",
     "--include", "ditto_cfg/*", "--include", "ditto_pytorch/*",
     "--local-dir", str(D / "models" / "ditto")])
run([hf, "download", "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
     "--local-dir", str(D / "models" / "qwen3-tts-1.7B")])

info("Vérification du GPU vu depuis WSL")
py = str(D / ".venv" / "bin" / "python")
run([py, "-c",
     "import torch; print('CUDA :', torch.cuda.is_available(), '-', "
     "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'AUCUN GPU — arrêtez-vous ici')"])

info("Fichiers privés à copier depuis Windows (jamais dans Git) :")
os.chdir(D)
for f in [".env", "assets/visage_pierre.jpg", "models/ref_pierre.wav",
          "models/question_test_16k.wav",
          "connaissances/iopool-agent-reference-docs/README.md"]:
    etat = "ok     " if Path(f).exists() else "MANQUE "
    print(f"   {etat} {f}")
print("Exemple : cp /mnt/c/Users/<vous>/Desktop/ref_pierre.wav models/")
print("Ensuite :  bash ~/iopool-aquaphone-MI/windows/lancer.sh   (serveur complet)")
print("           .venv/bin/python labo2/clips_local.py \"Bonjour.\"   (sans LLM)")
