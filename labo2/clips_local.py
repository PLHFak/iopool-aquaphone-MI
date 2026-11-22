"""Aquaphone-MI : clip local SANS LLM — texte → voix clonée (Qwen3-TTS) → visage animé (Ditto) → MP4.

Ni reconnaissance vocale, ni Claude, ni serveur web, ni mot de passe : juste la synthèse
et l'animation, pour tester la voix et le visage sur le PC Windows/WSL2 (RTX 4090) ou le pod.
Réutilise tel quel le Visage/parler() de labo/serveur.py (aucun fichier existant modifié) ;
le seul flux GPU interactif (Whisper en service, visage en flux) n'est pas perturbé tant
que le serveur n'est pas lancé en même temps.

Usage (depuis la racine du dépôt) :
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python labo2/clips_local.py "Une eau verte après un chlore choc, ça veut souvent dire que le pH est trop haut. Votre sonde affiche quoi ?"
    .venv/bin/python labo2/clips_local.py --fichier texte.txt          # plusieurs paragraphes
    .venv/bin/python labo2/clips_local.py "Bonjour !" --seuil 0.80      # contrôle par empreinte (une fois calé)
Sans --seuil : un tirage par phrase, graine fixe (reproductible).

Sorties : labo2/clips/<horodatage>_<n>.mp4 (+ .wav si ffmpeg manque) et labo2/clips/notes.json
Prérequis : windows/installer.sh déjà exécuté (modèles téléchargés), assets/visage_pierre.jpg et
models/ref_pierre.wav présents. PAS besoin d'ANTHROPIC_API_KEY ni de .env (sauf réglages).
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
sys.path.insert(0, str(RACINE / "labo"))

ap = argparse.ArgumentParser()
ap.add_argument("texte", nargs="?", default="")
ap.add_argument("--fichier", default="", help="fichier texte à lire (une réponse par paragraphe)")
ap.add_argument("--seuil", type=float, default=None, help="contrôle par empreinte (ex. 0.80), sinon un tirage par phrase")
ap.add_argument("--sortie", default=str(RACINE / "models" / "qwen3-tts-1.7B"), help="dossier du modèle de voix")
args = ap.parse_args()
if args.fichier:
    reponses = [p.strip() for p in Path(args.fichier).read_text(encoding="utf-8").split("\n\n") if p.strip()]
elif args.texte:
    reponses = [args.texte]
else:
    sys.exit("Donner un texte entre guillemets ou --fichier fichier.txt")

os.environ.setdefault("AQUAPHONE_PERSONA", "pierre")
import serveur  # noqa: E402  (charge la config, PAS le service : Whisper/Claude restent None)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402

# ─── Voix ────────────────────────────────────────────────────────────────────
print("Chargement de la voix (Qwen3-TTS 1.7B)…", flush=True)
t0 = time.perf_counter()
serveur.TTS = FasterQwen3TTS.from_pretrained(args.sortie, device="cuda:0")
if os.environ.get("AQUAPHONE_PREDICTOR"):  # ex. « 0.5,25 »
    tt, kk = os.environ["AQUAPHONE_PREDICTOR"].split(",")
    pg = serveur.TTS.predictor_graph
    pg.temperature, pg.top_k = float(tt), int(kk)
    pg.capture(num_warmup=3)
    print(f"Code predictor recapturé : {pg.temperature}, {pg.top_k}", flush=True)
print(f"Voix prête ({time.perf_counter() - t0:.0f} s)", flush=True)

if args.seuil is not None:
    ref, sr = sf.read(serveur.REF_AUDIO, dtype="float32")
    from scipy.signal import resample_poly  # noqa: E402
    if ref.ndim > 1:
        ref = ref.mean(1)
    if sr != 24000:
        ref = resample_poly(ref, 24000, sr).astype(np.float32)
    serveur.E_REF = serveur.empreinte_voix(ref)
    serveur.SEUIL_EMPREINTE = args.seuil
    print(f"Contrôle par empreinte : seuil {args.seuil}", flush=True)

# ─── Visage (identique à serveur.charger, sans le reste) ──────────────────────
print("Chargement du visage (Ditto)…", flush=True)
t0 = time.perf_counter()
if not serveur.SOURCE.exists():
    import cv2  # noqa: E402
    img = cv2.imread(str(RACINE / "assets" / "visage_pierre.jpg"))
    h, w = img.shape[:2]
    cv2.imwrite(str(serveur.SOURCE), cv2.resize(img, (round(w * serveur.HAUTEUR_SOURCE / h), serveur.HAUTEUR_SOURCE),
                                                interpolation=cv2.INTER_AREA))
sys.path.insert(0, str(RACINE / "third_party" / "ditto"))
from ditto_sdk import StreamSDK  # noqa: E402  (même import que serveur.py)

sdk = StreamSDK(str(RACINE / "models/ditto/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"),
                str(RACINE / "models/ditto/ditto_pytorch"))
n_fils = []
fil = threading.Thread(target=lambda: n_fils.append(torch.get_num_threads()))
fil.start()
fil.join()
torch.set_num_threads(n_fils[0])
sdk.warp_f3d.warp_net.model.compile()
sdk.decode_f3d.decoder.model.compile()
f3d = np.zeros((1, 32, 16, 64, 64), dtype=np.float32)
kp = np.zeros((1, 21, 3), dtype=np.float32)
sdk.decode_f3d(sdk.warp_f3d(f3d, kp, kp))
sortie_sdk = RACINE / "models/sorties_visage"
sortie_sdk.mkdir(parents=True, exist_ok=True)
sdk.setup(str(serveur.SOURCE), str(sortie_sdk / "local.mp4"), online_mode=True, overlap_v2=70,
          sampling_timesteps=serveur.PAS_DIFFUSION)
try:
    sdk.writer.close() if hasattr(sdk.writer, "close") else None
except Exception:
    pass
serveur.VISAGE = serveur.Visage(sdk)
serveur.VISAGE.commencer(None)  # préchauffage : 2 s de silence
serveur.VISAGE.pousser(np.zeros(2 * 24000, np.float32))
serveur.VISAGE.terminer()
limite = time.perf_counter() + 60
while ((serveur.VISAGE.n_nourries < 50 or serveur.VISAGE.n_emises < serveur.VISAGE.n_nourries - 25)
       and time.perf_counter() < limite):
    time.sleep(0.1)
time.sleep(0.5)
print(f"Visage prêt ({time.perf_counter() - t0:.0f} s)", flush=True)


# ─── Recoit les paquets (pcm 40 ms + jpeg) et fabrique un MP4 ─────────────────
class Enregistreur:
    def __init__(self):
        self.paquets = []
        self.fini = threading.Event()

    def image(self, rid, pcm, jpeg):
        self.paquets.append((pcm, jpeg))

    def fin(self, rid):
        self.fini.set()


def clip(phrase, fichier):
    if serveur.E_REF is not None:
        audio = np.concatenate([np.asarray(m, np.float32).reshape(-1) for m, _, _ in serveur.parler_controle(phrase)])
    else:
        torch.manual_seed(1234)
        audio = np.concatenate([np.asarray(m, np.float32).reshape(-1) for m, _, _ in serveur.parler(phrase)])
    fort = np.flatnonzero(np.abs(audio) > 0.02 * np.abs(audio).max()) if len(audio) else []
    if len(fort):
        audio = audio[max(0, fort[0] - 1200):fort[-1] + 3600]
    enreg = Enregistreur()
    ancienne = serveur.VISAGE.sortie
    serveur.VISAGE.sortie = enreg
    rid = f"local:{fichier.name}"
    serveur.VISAGE.commencer(rid)
    serveur.VISAGE.pousser(audio)
    serveur.VISAGE.terminer()
    enreg.fini.wait(120)
    serveur.VISAGE.sortie = ancienne
    for cle in ("t_premiere", "t_derniere", "n_images"):
        getattr(serveur.VISAGE, cle).pop(rid, None)
    if not enreg.paquets:
        return 0
    # son : pcm int16 24 kHz bout à bout ; images : jpegs 25 i/s produits par Ditto
    wav = fichier.with_suffix(".wav")
    pcm = np.concatenate([np.frombuffer(p, "<i2") if isinstance(p, (bytes, bytearray))
                          else np.asarray(p, dtype="<i2") for p, _ in enreg.paquets])
    sf.write(wav, pcm, 24000, subtype="PCM_16")
    dossier_tmp = fichier.parent / (fichier.stem + "_img")
    dossier_tmp.mkdir(exist_ok=True)
    for i, (_, jpeg) in enumerate(enreg.paquets):
        (dossier_tmp / f"{i:05d}.jpg").write_bytes(jpeg)
    lst = dossier_tmp / "list.txt"
    lst.write_text("".join(f"file '{i:05d}.jpg'\nduration 0.04\n" for i in range(len(enreg.paquets)))
                   + f"file '{len(enreg.paquets) - 1:05d}.jpg'\n")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-i", str(wav), "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(fichier)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  ffmpeg indisponible ({e}) : images dans {dossier_tmp}, son = {wav}", flush=True)
    return len(enreg.paquets)


# ─── Boucle : chaque paragraphe/phrase → un clip ──────────────────────────────
dossier = ICI / "clips"
dossier.mkdir(exist_ok=True)
horodatage = datetime.datetime.now().strftime("%H%M%S")
notes = []
for r, reponse in enumerate(reponses, 1):
    phrases = [p.strip() for p in re.findall(r".+?[.!?…](?=\s|$)", reponse) if len(p.strip()) > 1] or [reponse]
    for n, phrase in enumerate(phrases, 1):
        f = dossier / f"{horodatage}_{n:02d}.mp4"
        print(f"Clip {r}.{n} : « {phrase[:60]}… »", flush=True)
        t0 = time.perf_counter()
        nb = clip(phrase, f)
        notes.append({"clip": f.name, "phrase": phrase, "images": nb,
                      "duree_s": round(time.perf_counter() - t0, 1)})
        print(f"  → {f.name} ({nb} images, {time.perf_counter() - t0:.1f} s)", flush=True)
(dossier / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
print("Terminé :", len(notes), "clips dans", dossier, flush=True)
