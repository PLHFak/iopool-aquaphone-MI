"""Briques 5 et 6 — Pipeline de bout en bout + page web de test.

Navigateur ──(micro, PCM 16 kHz, WebSocket)──▶ VAD Silero ▶ faster-whisper ▶ Sonnet (flux, par phrase)
▶ Qwen3-TTS voix clonée (flux, 24 kHz) ▶ Ditto (flux, 25 images/s)
──(paquets « 1 image JPEG + ses 40 ms d'audio »)──▶ navigateur (son et image synchronisés)

Sur le PC, le serveur n'écoute que sur 127.0.0.1 ; l'accès depuis le Mac/iPhone passe par `tailscale serve`
(HTTPS, obligatoire pour le micro dans Safari), sans rien ouvrir dans le pare-feu. Sur RunPod
(AQUAPHONE_HOTE=0.0.0.0, proxy HTTPS public), la page est protégée par mot de passe (labo/acces.py).

Usage : python labo/serveur.py      puis ouvrir https://<nom-du-pc>.<tailnet>.ts.net/
Journal des délais : labo/journaux/pipeline.jsonl
"""
import asyncio
import collections
import itertools
import json
import os
import queue
import re
import secrets
import struct
import subprocess
import tempfile
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_win  # noqa: F401,E402  (DLL CUDA 12 pour CTranslate2 et onnxruntime)

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import anthropic  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402
import torch  # noqa: E402
from aiohttp import WSMsgType, web  # noqa: E402
from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.vad import get_assets_path  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
import soundfile as sf  # noqa: E402

import cerveau  # noqa: E402  (charge aussi .env)
import acces  # noqa: E402
import personas  # noqa: E402
import piscines  # noqa: E402
import attente  # noqa: E402
import contexte  # noqa: E402
import mode_labo  # noqa: E402
import vitesse  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "third_party" / "ditto"))
from stream_pipeline_online import StreamSDK  # noqa: E402

# ─── Réglages ────────────────────────────────────────────────────────────────
PORT = int(os.environ.get("AQUAPHONE_PORT", "8765"))
HOTE = os.environ.get("AQUAPHONE_HOTE", "127.0.0.1")  # 0.0.0.0 sur RunPod (adresse publique du proxy)
WEB = Path(__file__).resolve().parent / "web"
JOURNAL = Path(__file__).resolve().parent / "journaux" / "pipeline.jsonl"

# Détection de fin de parole (Silero, morceaux de 32 ms)
VAD_SEUIL = 0.5
SILENCE_FIN = float(os.environ.get("AQUAPHONE_SILENCE", "0.5"))  # silence qui clôt la phrase (s)
PAROLE_MIN = 0.3  # en dessous : bruit, ignoré (s)
PRE_ROLL = 0.3  # audio gardé avant le début détecté (s)
GARDE_ECHO = 0.6  # s de micro ignoré après la fin de lecture signalée par la page (anti-écho)
ATTENTE_SUITE = 1.2  # phrase transcrite inachevée (« je voudrais savoir... ») : délai pour la reprise (s)
PROMPT_STT = "Aquaphone. Un client parle à l'assistant de son pisciniste : piscine, chlore choc, pH, iopool."

# Personnage (visage + voix) : AQUAPHONE_PERSONA=pierre (défaut) ou un dossier personas/<nom>/
# contenant fiche.json (texte_reference), visage.jpg et voix.wav (jamais commités).
PERSONA = os.environ.get("AQUAPHONE_PERSONA") or personas.actif()  # choix du menu (personas/actif.txt)
REF_AUDIO = str(RACINE / "models" / "ref_pierre.wav")
REF_TEXTE = (
    "Bonjour, je m'appelle Pierre L'Hoest, je suis votre pisciniste, ici c'est mon clone "
    "qui vous répond, mais je pense que je peux vous répondre correctement, surtout si vous "
    "avez une sonde iopool dans votre piscine, je peux intégrer ces données dans notre "
    "discussion et ce sera bien."
)
TTS_CHUNK = 4
VOIX_TEMPERATURE = float(os.environ.get("AQUAPHONE_VOIX_TEMPERATURE", "0.6"))
VOIX_GRAINE = 1234
# Mode de clonage (labo2/test_etages.py, test_rejeu.py, choix de Pierre le 20/11) :
#  « empreinte » (défaut) : le modèle ne reçoit que l'empreinte du locuteur (xvec_only) → timbre stable quel que
#     soit le texte (ressemblance 0,984 en moyenne contre 0,937, hauteur 133-152 Hz sur les phrases qui dérivaient) ;
#  « continuation » : ancien mode, le modèle continue l'enregistrement de référence (l'intonation du texte
#     emporte le timbre : questions à 200 Hz et plus).
VOIX_MODE = os.environ.get("AQUAPHONE_VOIX_MODE", "empreinte")
VOIX_XVEC = VOIX_MODE == "empreinte"
VOIX_VITESSE = float(os.environ.get("AQUAPHONE_VOIX_VITESSE", "1.15"))  # débit (×1,15 choisi par Pierre le 20/11)
FICHIER_VITESSE = RACINE / "labo" / "journaux" / "vitesse_voix.txt"  # réglage du menu ☰, gardé au redémarrage
try:
    VOIX_VITESSE = float(FICHIER_VITESSE.read_text(encoding="utf-8"))
except (OSError, ValueError):
    pass
VITESSE_CLIPS = VOIX_VITESSE  # vitesse des phrases d'attente (calculées au démarrage)
# Contrôle de la voix (labo2/test_controle_voix.py) : chaque phrase est un tirage au hasard et ≈ 1 sur 8
# part loin de la référence (timbre plus aigu, « voix de femme »). La 1re seconde suffit à le voir
# (10 ratés sur 10 détectés, 3 fausses alertes sur 80) : on la garde en réserve, on mesure la hauteur, et
# si elle sort de la plage de la référence on abandonne et on refait avec une autre graine.
CONTROLE_S = 1.2           # audio mesuré avant de jouer la phrase
CONTROLE_ESSAIS = 3        # essais au plus ; le dernier est joué quoi qu'il arrive
CONTROLE_BORNES = (0.73, 1.34)  # plage acceptée, en multiples de la hauteur médiane de la référence
REF_HAUTEUR = None         # mesurée au chargement
# Contrôle v2 par EMPREINTE du locuteur (brief labo2/BRIEF_CORRECTION_VOIX.md, mesures : labo2/essai_empreinte.py).
# Inactif tant que le seuil n'est pas calé sur les mesures du pod : AQUAPHONE_SEUIL_EMPREINTE=0.xx ; sans seuil,
# le contrôle v1 (hauteur) reste en place, comme avant. Voix de Pierre seulement (le seuil est calé sur sa référence).
SEUIL_EMPREINTE = float(os.environ["AQUAPHONE_SEUIL_EMPREINTE"]) if os.environ.get("AQUAPHONE_SEUIL_EMPREINTE") else None
CONTROLE_V2_ESSAIS = 5                 # le meilleur des 5 est joué si aucun n'atteint le seuil (on ne peut pas ne pas répondre)
CONTROLE_V2_BORNES = (0.80, 1.25)      # hauteur : garde-fou secondaire, en multiples de la référence
CONTROLE_V2_MAX_CAR = 120              # au-delà (≈ 8 s de parole), on reste en flux avec le contrôle v1 : pas d'attente de fin
PREDICTOR = os.environ.get("AQUAPHONE_PREDICTOR", "")  # « T,K » : hasard réduit du code predictor (ex. « 0.5,25 »), vide = bibliothèque
E_REF = None                           # empreinte de la référence, calculée au chargement si le seuil est fixé
VOIX_STATS = {"phrases": 0, "refaites": 0, "essais_ecartes": 0,  # depuis le démarrage (menu ☰ → Version)
              "similarite_moyenne": 0.0, "meilleur_essai": {}, "sauvees_par_hauteur": 0, "sous_le_seuil": 0}
_SIM = {"somme": 0.0, "n": 0}


def _version():
    """Empreinte du code réellement chargé (le dossier du pod n'est pas un dépôt Git à jour) + commit si connu."""
    import hashlib
    h = hashlib.sha1()
    for f in sorted((RACINE / "labo").glob("*.py")) + [RACINE / "labo" / "web" / "index.html"]:
        h.update(f.read_bytes())
    fichier = RACINE / "labo" / "version.txt"  # écrit au déploiement (commit Git), absent sinon
    commit = fichier.read_text(encoding="utf-8").strip() if fichier.exists() else "inconnu"
    return {"code": h.hexdigest()[:7], "commit": commit, "demarre": time.strftime("%d/%m %H:%M:%S")}


VERSION = _version()


def hauteur_mediane(a, sr=24000):
    """Hauteur médiane de la voix (Hz) par autocorrélation sur des fenêtres de 85 ms ; 0 si inaudible."""
    v = []
    for i in range(0, len(a) - 2048, sr // 10):
        x = a[i:i + 2048] - a[i:i + 2048].mean()
        if np.sqrt((x ** 2).mean()) < 0.01:
            continue
        c = np.correlate(x, x, "full")[len(x) - 1:]
        lo, hi = sr // 400, sr // 60
        k = lo + int(np.argmax(c[lo:hi]))
        if c[k] > 0.3 * c[0]:
            v.append(sr / k)
    return float(np.median(v)) if v else 0.0


def parler(texte, graine=VOIX_GRAINE):
    """Synthèse clonée en flux (un seul tirage), appelée dans le fil EXEC."""
    torch.manual_seed(graine)
    return TTS.generate_voice_clone_streaming(text=texte, language="French", ref_audio=REF_AUDIO,
                                              ref_text=REF_TEXTE, chunk_size=TTS_CHUNK,
                                              temperature=VOIX_TEMPERATURE, top_k=30, xvec_only=VOIX_XVEC)


def empreinte_voix(a):
    """Empreinte du locuteur (encodeur de Qwen3-TTS lui-même), normalisée. Fil EXEC, entre deux générations."""
    e = TTS.model.model.extract_speaker_embedding(audio=a.astype(np.float32), sr=24000).float().reshape(-1)
    return torch.nn.functional.normalize(e, dim=0)


def parler_controle(texte):
    """Voix contrôlée puis accélérée (VOIX_VITESSE, sans changer la hauteur, en flux : labo/vitesse.py)."""
    yield from vitesse.accelerer(_parler_controle(texte), VOIX_VITESSE)


def _parler_controle(texte):
    """Voix contrôlée. Phrases courtes avec seuil d'empreinte fixé : v2 (empreinte) ; sinon v1 (hauteur, en flux)."""
    if E_REF is not None and len(texte) <= CONTROLE_V2_MAX_CAR:
        yield from parler_controle_v2(texte)
    else:
        yield from parler_controle_v1(texte)


def parler_controle_v2(texte):
    """Jusqu'à 5 tirages (graines 1234+k), chacun mené jusqu'au bout dans le fil EXEC (pas de close() à mi-course :
    pas d'état GPU douteux) puis noté par similarité d'empreinte avec la référence. Le 1er tirage au seuil est joué ;
    si aucun n'y arrive, le meilleur des 5. La hauteur est un garde-fou secondaire : un tirage n'est écarté que si
    l'empreinte ET la hauteur dérivent. Coût : la phrase n'est rendue qu'une fois générée (≈ sa durée × vitesse de la voix)."""
    VOIX_STATS["phrases"] += 1
    tirages = []  # (similarité, morceaux, sauvé par la hauteur)
    for essai in range(CONTROLE_V2_ESSAIS):
        morceaux = []
        for morceau, sr, _ in parler(texte, VOIX_GRAINE + essai):
            morceau = np.asarray(morceau, np.float32).reshape(-1)
            morceaux.append(morceau if sr == 24000 else resample_poly(morceau, 24000, sr).astype(np.float32))
        if not morceaux:
            continue
        audio = np.concatenate(morceaux)
        sim = float(torch.dot(empreinte_voix(audio), E_REF))
        h = hauteur_mediane(audio)
        hauteur_ok = h == 0.0 or CONTROLE_V2_BORNES[0] * REF_HAUTEUR <= h <= CONTROLE_V2_BORNES[1] * REF_HAUTEUR
        tirages.append((sim, morceaux, sim < SEUIL_EMPREINTE and hauteur_ok))
        if sim >= SEUIL_EMPREINTE or hauteur_ok:
            break
        VOIX_STATS["essais_ecartes"] += 1
        log(f"Voix : essai {essai + 1} écarté (empreinte {sim:.2f} < {SEUIL_EMPREINTE:.2f}, {h:.0f} Hz) « {texte[:40]} »")
    if not tirages:
        return
    if tirages[-1][0] >= SEUIL_EMPREINTE or tirages[-1][2]:
        rang = len(tirages) - 1  # le dernier tirage est celui qui a convenu
    else:
        rang = max(range(len(tirages)), key=lambda i: tirages[i][0])
        VOIX_STATS["sous_le_seuil"] += 1
        log(f"Voix : aucun essai au seuil, meilleur des {len(tirages)} joué (empreinte {tirages[rang][0]:.2f}) « {texte[:40]} »")
    sim, morceaux, sauve = tirages[rang]
    VOIX_STATS["sauvees_par_hauteur"] += int(sauve)
    if rang:
        VOIX_STATS["refaites"] += 1
    VOIX_STATS["meilleur_essai"][str(rang + 1)] = VOIX_STATS["meilleur_essai"].get(str(rang + 1), 0) + 1
    _SIM["somme"] += sim
    _SIM["n"] += 1
    VOIX_STATS["similarite_moyenne"] = round(_SIM["somme"] / _SIM["n"], 3)
    yield from morceaux


def parler_controle_v1(texte):
    """Comme parler(), en morceaux float32 à 24 kHz, mais la 1re seconde est vérifiée avant d'être rendue :
    hauteur hors de la plage de la référence → tirage abandonné et refait (autre graine)."""
    VOIX_STATS["phrases"] += 1
    for essai in range(CONTROLE_ESSAIS):
        flux = parler(texte, VOIX_GRAINE + essai)
        tete, duree = [], 0.0
        for morceau, sr, _ in flux:
            morceau = np.asarray(morceau, np.float32).reshape(-1)
            if sr != 24000:
                morceau = resample_poly(morceau, 24000, sr).astype(np.float32)
            tete.append(morceau)
            duree += len(morceau) / 24000
            if duree >= CONTROLE_S:
                break
        h = hauteur_mediane(np.concatenate(tete)) if tete else 0.0
        bon = REF_HAUTEUR is None or h == 0.0 or (
            CONTROLE_BORNES[0] * REF_HAUTEUR <= h <= CONTROLE_BORNES[1] * REF_HAUTEUR)
        if bon or essai == CONTROLE_ESSAIS - 1:
            if essai:
                VOIX_STATS["refaites"] += 1
                log(f"Voix : essai {essai + 1} retenu ({h:.0f} Hz, référence {REF_HAUTEUR:.0f} Hz) « {texte[:40]} »")
            yield from tete
            for morceau, sr, _ in flux:
                morceau = np.asarray(morceau, np.float32).reshape(-1)
                yield morceau if sr == 24000 else resample_poly(morceau, 24000, sr).astype(np.float32)
            return
        flux.close()
        VOIX_STATS["essais_ecartes"] += 1
        log(f"Voix : essai {essai + 1} écarté ({h:.0f} Hz, hors de {CONTROLE_BORNES[0] * REF_HAUTEUR:.0f}–"
            f"{CONTROLE_BORNES[1] * REF_HAUTEUR:.0f} Hz) « {texte[:40]} »")
TAILLE_VOIX = os.environ.get("AQUAPHONE_VOIX", "1.7B")  # 1.7B ou 0.6B
# Carte graphique de la voix et de la transcription. Le visage (Ditto) reste sur la carte 0.
# PC labo (1 carte) : 0. Machine à 2 cartes (RunPod) : 1 → la voix ne ralentit plus le visage.
GPU_VOIX = int(os.environ.get("AQUAPHONE_GPU_VOIX", "0"))

# Visage : 1 image = 40 ms = 960 échantillons à 24 kHz = 640 à 16 kHz
IPS = 25
ECH_IMAGE = 24000 // IPS
BLOC_IMAGES = 5
BLOC_24 = BLOC_IMAGES * ECH_IMAGE
FENETRE_16 = (3 + 5 + 2) * 640 + 80  # fenêtre HuBERT : 3 images passées, 5 courantes, 2 futures
PAS_16 = BLOC_IMAGES * 640
BLOCS_VIDANGE = 8  # silence ajouté en fin de réponse pour sortir les dernières images
AVANCE_MIN = 0.8  # s d'audio d'avance sous laquelle on insère un silence (évite les coupures)
LARGEUR_VIDEO = int(os.environ.get("AQUAPHONE_LARGEUR", "480"))  # largeur de la vidéo envoyée (px) ; 720 = HD
QUALITE_JPEG = int(os.environ.get("AQUAPHONE_JPEG", "84"))  # qualité JPEG cible (le visage bouge peu : haute qualité possible)
BUDGET_IMAGE = int(os.environ.get("AQUAPHONE_BUDGET_IMAGE", "40000"))  # compression adaptative : taille max d'une image (octets)
QUALITE_JPEG_MIN = 55  # plancher de qualité quand une image dépasse le budget
HAUTEUR_SOURCE = int(os.environ.get("AQUAPHONE_HAUTEUR", "720"))  # photo sur laquelle le visage est recollé
SOURCE = RACINE / "models" / f"visage_pierre_{HAUTEUR_SOURCE}.jpg"
if PERSONA != "pierre":
    _dossier = RACINE / "personas" / PERSONA
    _fiche = json.loads((_dossier / "fiche.json").read_text(encoding="utf-8"))
    REF_AUDIO, REF_TEXTE = str(_dossier / "voix.wav"), _fiche["texte_reference"]
    SOURCE = _dossier / f"visage_{HAUTEUR_SOURCE}.jpg"
    if not SOURCE.exists():  # photo réduite à la hauteur de travail (comme visage_pierre_720.jpg)
        _img = cv2.imread(str(_dossier / "visage.jpg"))
        _l = round(_img.shape[1] * HAUTEUR_SOURCE / _img.shape[0] / 2) * 2
        cv2.imwrite(str(SOURCE), cv2.resize(_img, (_l, HAUTEUR_SOURCE), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
PAS_DIFFUSION = int(os.environ.get("AQUAPHONE_PAS", "5"))  # étapes du modèle de mouvement (Ditto)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ─── Écoute : VAD Silero en flux ─────────────────────────────────────────────
class Vad:
    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            os.path.join(get_assets_path(), "silero_vad_v6.onnx"),
            providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.contexte = np.zeros(64, np.float32)

    def __call__(self, morceau):
        x = np.concatenate([self.contexte, morceau])[None].astype(np.float32)
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.contexte = morceau[-64:]
        return float(np.asarray(out).reshape(-1)[0])


class Ecoute:
    """Reçoit le micro par petits morceaux ; rend (audio de la phrase, instant de fin de parole, complète).

    Une phrase transcrite inachevée est mise « en suspens » : si le client reprend la parole,
    la suite y est recollée ; sinon on répond quand même au bout de ATTENTE_SUITE.
    """

    def __init__(self):
        self.vad = Vad()
        self.reset()

    def reset(self):
        self.vad.reset()
        self.reste = np.zeros(0, np.float32)
        self.avant = collections.deque(maxlen=int(PRE_ROLL * 16000 / 512))
        self.phrase = []
        self.n_parole = self.n_silence = 0
        self.t_fin = None
        self.en_suspens = None  # (audio, t_fin, instant de la mise en suspens)

    def suspendre(self, audio, t_fin):
        self.reset()
        self.en_suspens = (audio, t_fin, time.perf_counter())

    def _avec_suspens(self, audio, t_fin):
        if self.en_suspens is None:
            return audio, t_fin, False
        avant, self.en_suspens = self.en_suspens[0], None
        return np.concatenate([avant, np.zeros(int(0.2 * 16000), np.float32), audio]), t_fin, True

    @property
    def en_parole(self):
        return bool(self.phrase)

    def ajouter(self, pcm):
        self.reste = np.concatenate([self.reste, pcm])
        resultat = None
        while len(self.reste) >= 512:
            m, self.reste = self.reste[:512], self.reste[512:]
            p = self.vad(m)
            if not self.phrase:
                self.avant.append(m)
                if p >= VAD_SEUIL:
                    self.phrase, self.n_parole, self.n_silence = list(self.avant), 1, 0
                continue
            self.phrase.append(m)
            if p >= VAD_SEUIL - 0.15:
                self.n_parole += p >= VAD_SEUIL
                self.n_silence = 0
            else:
                if self.n_silence == 0:
                    self.t_fin = time.perf_counter()  # fin de parole (à confirmer par le silence)
                self.n_silence += 1
                if self.n_silence * 512 / 16000 >= SILENCE_FIN:
                    if self.n_parole * 512 / 16000 >= PAROLE_MIN:
                        resultat = self._avec_suspens(np.concatenate(self.phrase), self.t_fin)
                    self.phrase, self.n_parole, self.n_silence = [], 0, 0
                    self.avant.clear()
        if (resultat is None and self.en_suspens and not self.phrase
                and time.perf_counter() - self.en_suspens[2] > ATTENTE_SUITE):
            resultat = (self.en_suspens[0], self.en_suspens[1], True)  # pas de suite : on répond
            self.en_suspens = None
        return resultat


def _mots(texte):
    return re.findall(r"\w+", texte.lower())


# Phrases que Whisper invente sur du bruit : reprise de son invite, génériques de sous-titres.
HALLUCINATION = re.compile(r"client parle à l.assistant|assistant de son piscin|sous-titr|ST' ?501|amara\.org|"
                           r"merci d.avoir regardé|abonnez-vous", re.I)


def est_echo(texte, dit):
    """Vrai si la phrase entendue reprend surtout ce que l'assistant vient de dire (haut-parleur → micro)."""
    entendus = _mots(texte)
    if len(entendus) < 2 or not dit:
        return False
    prononces = set(_mots(" ".join(dit)))
    communs = sum(m in prononces for m in entendus)
    return communs / len(entendus) >= 0.7


# ─── Visage : Ditto alimenté en continu ──────────────────────────────────────
class Visage:
    """Alimente Ditto bloc par bloc (0,2 s) et associe chaque image produite à ses 40 ms d'audio.

    Ditto produit les images dans l'ordre où l'audio lui est donné : une file (audio, réponse)
    suffit à les apparier. Entre deux réponses, on n'alimente plus Ditto : le visage reste figé.
    """

    def __init__(self, sdk):
        self.sdk = sdk
        self.cond = threading.Condition()
        self.attente = np.zeros(0, np.float32)
        self.rid, self.actif, self.fini = None, False, False
        self.n_reponse = 0
        self.silences = 0
        self.t_premiere = {}
        self.t_derniere, self.n_images = {}, {}  # diagnostic : cadence réelle du visage par réponse
        self.fins = set()
        self.images = collections.deque()
        self.buf16 = np.zeros(3 * 640, np.float32)  # contexte initial (comme dans le test isolé)
        self.sortie = None
        self.n_nourries = self.n_emises = 0
        sdk.writer = self  # Ditto appelle self(image, fmt) depuis son fil « writer »
        threading.Thread(target=self._boucle, daemon=True, name="visage").start()

    # appelé par le fil de la réponse
    def commencer(self, rid):
        with self.cond:
            self.rid, self.actif, self.fini = rid, True, False
            self.n_reponse = self.silences = 0
            self.attente = np.zeros(0, np.float32)

    def pousser(self, audio24):
        with self.cond:
            self.attente = np.concatenate([self.attente, audio24])
            self.cond.notify()

    def terminer(self):
        with self.cond:
            self.fini = True
            self.cond.notify()

    def _avance(self):
        t0 = self.t_premiere.get(self.rid)
        if t0 is None:
            return float("inf")
        return self.n_reponse / IPS - (time.perf_counter() - t0)

    def _boucle(self):
        zeros = np.zeros(BLOC_24, np.float32)
        while True:
            with self.cond:
                while True:
                    if len(self.attente) >= BLOC_24:
                        bloc, self.attente, dernier = self.attente[:BLOC_24], self.attente[BLOC_24:], False
                        break
                    if self.actif and self.fini:
                        bloc = np.pad(self.attente, (0, BLOC_24 - len(self.attente)))
                        self.attente, self.actif, dernier = np.zeros(0, np.float32), False, True
                        break
                    if self.actif and self._avance() < AVANCE_MIN:
                        bloc, dernier = zeros, False  # la voix prend du retard : on comble
                        self.silences += 1
                        break
                    self.cond.wait(timeout=0.02)
                rid = self.rid
            try:
                self._nourrir(bloc, rid, False)
                if dernier:
                    for _ in range(BLOCS_VIDANGE):
                        self._nourrir(zeros, rid, True)
            except Exception:
                traceback.print_exc()

    def _nourrir(self, bloc24, rid, vidange):
        for f in range(BLOC_IMAGES):
            self.images.append((bloc24[f * ECH_IMAGE:(f + 1) * ECH_IMAGE], rid, vidange))
        if not vidange:
            self.n_reponse += BLOC_IMAGES
        self.n_nourries += BLOC_IMAGES
        self.buf16 = np.concatenate([self.buf16, resample_poly(bloc24, 2, 3).astype(np.float32)])
        while len(self.buf16) >= FENETRE_16:
            self.sdk.run_chunk(self.buf16[:FENETRE_16], (3, 5, 2))
            self.buf16 = self.buf16[PAS_16:]

    # appelé par Ditto (fil « writer »)
    def __call__(self, img, fmt="rgb"):
        self.n_emises += 1
        try:
            audio, rid, vidange = self.images.popleft()
        except IndexError:
            return
        if rid is None:
            return
        if vidange:
            if rid not in self.fins:
                self.fins.add(rid)
                if self.sortie:
                    self.sortie.fin(rid)
            return
        maintenant = time.perf_counter()
        self.t_premiere.setdefault(rid, maintenant)
        self.t_derniere[rid] = maintenant
        self.n_images[rid] = self.n_images.get(rid, 0) + 1
        h, w = img.shape[:2]
        petite = cv2.resize(img, (LARGEUR_VIDEO, round(h * LARGEUR_VIDEO / w)), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(petite, cv2.COLOR_RGB2BGR)
        _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, QUALITE_JPEG])
        # Compression adaptative (« VBR ») : le visage bouge peu, la plupart des images sont déjà légères
        # à la qualité cible ; seules les images riches (mouvement marqué, détails) dépassent le budget -->
        # on réduit alors leur qualité par crans pour préserver la fluidité sur internet.
        q = QUALITE_JPEG
        while jpeg.nbytes > BUDGET_IMAGE and q > QUALITE_JPEG_MIN:
            q -= 10
            _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        if self.sortie:
            self.sortie.image(rid, pcm, jpeg.tobytes())

    def close(self):
        pass


# ─── Session : une connexion WebSocket ───────────────────────────────────────
COMPTEUR = itertools.count(1)
EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu",  # STT + voix : un seul fil
                          initializer=lambda: torch.cuda.set_device(GPU_VOIX))
WHISPER = TTS = VISAGE = CLIENT = None
SESSIONS = set()
NOTES = collections.deque(maxlen=30)  # réponses écrites de Claude Code (mode labo), rejouées à la connexion
JETON = secrets.token_urlsafe(24)  # protège /note (la page est servie sur tout le tailnet)
FICHIER_JETON = JOURNAL.parent / "jeton_note.txt"
DOCUMENTS = ""
CLIPS = {}  # phrases d'attente pré-calculées : {phrase: [(pcm, jpeg), …]}


class Session:
    def __init__(self, ws, loop):
        self.ws, self.loop = ws, loop
        self.file = asyncio.Queue()
        self.historique = []
        self.mode = "client"  # « labo » après « Aquaphone » (voir mode_labo.py)
        self.ecoute = Ecoute()
        self.occupe = False
        self.parole_signalee = False
        self.retenu = None  # micro reçu pendant la transcription (rejoué si la phrase est inachevée)
        self.sourd_jusqua = 0.0  # anti-écho : micro ignoré jusqu'à cet instant
        self.t_pret = 0.0  # fin de la dernière lecture côté page
        self.dit_recemment = collections.deque(maxlen=12)  # dernières phrases prononcées (filtre d'écho)
        self.attentes_recentes = attente.nouveaux_recents()  # pas de répétition sur les 3 derniers tours
        self.mesures = {}

    # ─ fil de la boucle asyncio ─
    def micro(self, pcm):
        if time.perf_counter() < self.sourd_jusqua:
            return
        if self.occupe:
            if self.retenu is not None:
                self.retenu.append(pcm)
            return  # semi-duplex : on n'écoute pas pendant que l'assistant parle
        res = self.ecoute.ajouter(pcm)
        if self.ecoute.en_parole and not self.parole_signalee:
            self.parole_signalee = True
            self.envoyer({"type": "etat", "etat": "entendu"})
            # pendant que le client parle : ouvrir la connexion HTTPS vers l'API (−0,5 s au 1er appel)
            self.loop.run_in_executor(None, rechauffer_claude)
        if res:
            audio, t_fin, complet = res
            log(f"Fin de parole détectée ({len(audio) / 16000:.1f} s d'audio)")
            self.occupe, self.retenu = True, []
            self.envoyer({"type": "etat", "etat": "reflexion"})
            self.loop.run_in_executor(EXEC, self.traiter, audio, t_fin, None, complet)

    def suspendre(self, audio, t_fin):
        retenu, self.retenu = self.retenu or [], None
        self.ecoute.suspendre(audio, t_fin)
        self.parole_signalee = False
        self.occupe = False
        self.envoyer({"type": "etat", "etat": "ecoute"})
        for pcm in retenu:  # ce que le client a dit pendant la transcription
            self.micro(pcm)

    def ne_plus_retenir(self):
        self.retenu = None

    def envoyer(self, msg):  # depuis n'importe quel fil
        self.loop.call_soon_threadsafe(self.file.put_nowait, msg)

    async def expediteur(self):
        while (msg := await self.file.get()) is not None:
            try:
                if isinstance(msg, bytes):
                    await self.ws.send_bytes(msg)
                else:
                    await self.ws.send_str(json.dumps(msg, ensure_ascii=False))
            except (ConnectionResetError, RuntimeError):
                break

    # sortie du visage
    def image(self, rid, pcm, jpeg):
        self.envoyer(struct.pack("<BBHII", 1, 0, len(pcm) // 2, rid, 0) + pcm + jpeg)

    def fin(self, rid):
        self.envoyer({"type": "fin", "rid": rid})

    def traiter(self, audio, t_fin, texte=None, complet=True):
        """Fil « gpu » : transcription → Claude (fil séparé) → voix → visage."""
        rid = next(COMPTEUR)
        m = {"rid": rid, "t_fin": t_fin, "t_decision": time.perf_counter()}
        self.mesures[rid] = m
        demarre = False
        try:
            if texte is None:
                segments, _ = WHISPER.transcribe(audio, language="fr", beam_size=1, vad_filter=False,
                                                 initial_prompt=PROMPT_STT, condition_on_previous_text=False)
                # segments jugés « sans parole » par Whisper (bruit, télé lointaine) écartés
                texte = " ".join(s.text.strip() for s in segments
                                 if not (s.no_speech_prob > 0.6 and s.avg_logprob < -0.8)).strip()
                if HALLUCINATION.search(texte):
                    log(f"Transcription inventée par Whisper ignorée : « {texte} »")
                    texte = ""
                m["duree_parole"] = len(audio) / 16000
            m["t_stt"] = time.perf_counter()
            if not complet and texte.endswith(("...", "…")):
                log(f"Phrase inachevée, j'attends la suite : « {texte} »")
                self.mesures.pop(rid, None)
                self.loop.call_soon_threadsafe(self.suspendre, audio, t_fin)
                return
            self.loop.call_soon_threadsafe(self.ne_plus_retenir)
            m["question"] = texte
            log(f"Réponse {rid} — question : « {texte} »")
            noter_activite()
            if not texte:
                self.envoyer({"type": "etat", "etat": "ecoute"})
                self.reprendre()
                return
            debut_parole = t_fin - len(audio) / 16000 if audio is not None else 0.0
            if texte and debut_parole < self.t_pret + 2.0 and est_echo(texte, self.dit_recemment):
                log(f"Écho de la réponse ignoré : « {texte} »")
                self.mesures.pop(rid, None)
                self.reprendre()
                self.envoyer({"type": "etat", "etat": "ecoute"})
                return
            ancien = self.mode
            self.mode, texte = mode_labo.detecter(texte, self.mode)
            if self.mode != ancien:
                log(f"Passage en mode {self.mode}")
                self.envoyer({"type": "mode", "mode": self.mode})
            if self.mode == "labo" or not texte:
                # Mode labo : ni voix, ni visage, ni Claude — on note pour Claude Code et on réécoute.
                if self.mode == "labo" and texte:
                    mode_labo.noter_demande(texte)
                    log(f"Demande pour Claude Code : « {texte} »")
                if texte:
                    self.envoyer({"type": "transcription", "texte": texte, "mode": self.mode})
                self.mesures.pop(rid, None)
                self.reprendre()
                self.envoyer({"type": "etat", "etat": "ecoute"})
                return
            m["mode"] = self.mode
            self.envoyer({"type": "transcription", "texte": texte, "mode": self.mode})
            self.historique.append({"role": "user", "content": texte})
            # Phrase d'attente : clip tout prêt (voix + visage), envoyé avant que Claude ait répondu.
            phrase = attente.choisir(texte, self.attentes_recentes) if CLIPS else None
            if phrase in CLIPS:
                m["attente"] = phrase
                self.dit_recemment.append(phrase)
                for pcm, jpeg in CLIPS[phrase]:
                    self.envoyer(struct.pack("<BBHII", 1, 1, len(pcm) // 2, rid, 0) + pcm + jpeg)
                log(f"Phrase d'attente : « {phrase} »")
            VISAGE.sortie = self
            VISAGE.commencer(rid)
            demarre = True

            phrases, mc = queue.Queue(), {}

            def penser():
                try:
                    for p in cerveau.phrases_en_flux(CLIENT, list(self.historique), mesures=mc,
                                                     documents=DOCUMENTS, piscine=piscines.texte()):
                        m.setdefault("t_phrase1", time.perf_counter())
                        phrases.put(p)
                except Exception as e:  # transmis au fil gpu
                    phrases.put(e)
                finally:
                    phrases.put(None)

            threading.Thread(target=penser, daemon=True, name="claude").start()
            reponse = []
            while (p := phrases.get()) is not None:
                if isinstance(p, Exception):
                    raise p
                reponse.append(p)
                self.dit_recemment.append(p)
                self.envoyer({"type": "phrase", "texte": p})
                t_voix = time.perf_counter()
                for morceau in parler_controle(p):
                    m.setdefault("t_son1", time.perf_counter())
                    m["audio_s"] = m.get("audio_s", 0) + len(morceau) / 24000
                    VISAGE.pousser(morceau)
                m["calcul_voix_s"] = m.get("calcul_voix_s", 0) + time.perf_counter() - t_voix
            m["t_voix_finie"] = time.perf_counter()
            if "usage" in mc:
                u = mc["usage"]
                log(f"Claude : {u.input_tokens} tokens hors cache, {u.cache_read_input_tokens} lus en cache, "
                    f"{u.cache_creation_input_tokens} écrits en cache")
            m["reponse"] = " ".join(reponse)
            self.historique.append({"role": "assistant", "content": m["reponse"]})
        except Exception as e:
            traceback.print_exc()
            if self.historique and self.historique[-1]["role"] == "user" and self.mode == "client":
                self.historique.pop()
            self.envoyer({"type": "erreur", "texte": f"{type(e).__name__} : {e}"})
            if not demarre:
                self.reprendre()
        finally:
            if demarre:
                VISAGE.terminer()  # → silence de vidange → message « fin » → le navigateur répond « pret »

    def reprendre(self):
        self.ecoute.reset()
        self.parole_signalee = False
        self.retenu = None
        self.occupe = False

    def cloturer(self, rid, trous=None):
        m = self.mesures.pop(rid, None)
        if not m or "t_stt" not in m:
            return
        # Diagnostic des coupures : la voix et le visage tiennent-ils le temps réel ?
        n_img = VISAGE.n_images.pop(rid, 0)
        t1, t2 = VISAGE.t_premiere.get(rid), VISAGE.t_derniere.pop(rid, None)
        ips = n_img / (t2 - t1) if t1 and t2 and t2 > t1 else 0
        audio_s = m.get("audio_s", 0)
        vitesse_voix = audio_s / m["calcul_voix_s"] if m.get("calcul_voix_s") else 0
        trous = trous or []
        log(f"Diagnostic {rid} : réponse {audio_s:.1f} s | voix x{vitesse_voix:.1f} temps réel | "
            f"visage {ips:.1f} images/s (25 nécessaires) | coupures chez le client : {len(trous)} "
            f"{trous if trous else ''}")
        m["diag"] = {"audio_s": round(audio_s, 1), "voix_x_temps_reel": round(vitesse_voix, 2),
                     "visage_ips": round(ips, 1), "coupures": len(trous),
                     "coupures_ms": sum(t[1] for t in trous), "trous": trous}
        t_image = VISAGE.t_premiere.pop(rid, None)

        def ms(a, b):
            return round((m[b] - m[a]) * 1000) if a in m and b in m else None

        if t_image:
            m["t_image1"] = t_image
        res = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": m.get("question"),
            "reponse": m.get("reponse"),
            "duree_parole_s": round(m.get("duree_parole", 0), 2),
            "attente_silence_vad": ms("t_fin", "t_decision"),
            "transcription": ms("t_decision", "t_stt"),
            "claude_1re_phrase": ms("t_stt", "t_phrase1"),
            "voix_1er_son": ms("t_phrase1", "t_son1"),
            "visage_1re_image": ms("t_son1", "t_image1"),
            "reseau_et_tampon": ms("t_image1", "t_lecture"),
            "total_serveur": ms("t_fin", "t_image1"),
            "total_percu": ms("t_fin", "t_lecture"),
            "attente": m.get("attente"),
            "premier_son_percu": ms("t_fin", "t_lecture_attente") or ms("t_fin", "t_lecture"),
            "silences_inseres": VISAGE.silences,
            "mode": m.get("mode", "client"),
            **m.get("diag", {}),
        }
        log(f"Images données à Ditto {VISAGE.n_nourries}, produites {VISAGE.n_emises}")
        log("Délais (ms) :", {k: v for k, v in res.items() if k not in ("question", "reponse", "date")})
        JOURNAL.parent.mkdir(exist_ok=True)
        with open(JOURNAL, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        self.envoyer({"type": "mesures", **res})


def rechauffer_claude():
    try:
        CLIENT.models.list(limit=1)
    except Exception:
        pass


async def note(request):
    """Claude Code publie une réponse écrite : POST /note avec l'en-tête X-Jeton (voir labo/note.py)."""
    if request.headers.get("X-Jeton") != JETON:
        raise web.HTTPForbidden()
    texte = (await request.text()).strip()
    if texte:
        n = {"type": "note", "heure": time.strftime("%H:%M"), "texte": texte}
        NOTES.append(n)
        for s in list(SESSIONS):
            s.envoyer(n)
        log(f"Note de Claude Code affichée sur {len(SESSIONS)} page(s)")
    return web.Response(text="ok")


async def liste_personas(request):
    return web.json_response({"personas": personas.liste(PERSONA), "texte_a_lire": personas.TEXTE_A_LIRE,
                              "consentement": personas.CONSENTEMENT})


async def photo_persona(request):
    chemin = personas.photo(request.match_info["nom"])
    if request.match_info["nom"] not in [p["nom"] for p in personas.liste(PERSONA)] or not chemin.exists():
        raise web.HTTPNotFound()
    return web.FileResponse(chemin, headers={"Cache-Control": "no-store"})


async def deposer_persona(request):
    """Dépôt photo + voix dans une place libre (accord obligatoire)."""
    nom = request.match_info["nom"]
    donnees = await request.post()
    if donnees.get("accord") != "oui":
        return web.json_response({"erreur": "Accord obligatoire."}, status=400)
    image, voix = donnees.get("photo"), donnees.get("voix")
    if not hasattr(image, "file") or not hasattr(voix, "file"):
        return web.json_response({"erreur": "Photo et voix obligatoires."}, status=400)

    def transcrire(chemin):
        segments, _ = WHISPER.transcribe(chemin, language="fr", beam_size=5, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()

    try:
        texte = await asyncio.get_running_loop().run_in_executor(
            EXEC, personas.enregistrer, nom, str(donnees.get("prenom", "")), image.file.read(), voix.file.read(),
            transcrire)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        return web.json_response({"erreur": str(e)}, status=400)
    log(f"Personnage déposé dans {nom} (accord donné) : « {texte} »")
    return web.json_response({"ok": True, "texte": texte})


async def choisir_persona(request):
    """Change de personnage : écrit le choix puis redémarre le serveur (relancé par runpod/boucle.sh)."""
    nom = (await request.json()).get("nom", "")
    if not personas.pret(nom):
        return web.json_response({"erreur": "Personnage incomplet."}, status=400)
    personas.choisir(nom)
    log(f"Changement de personnage → {nom} : redémarrage")
    for s in list(SESSIONS):
        s.envoyer({"type": "redemarrage", "nom": nom})
    asyncio.get_running_loop().call_later(1.0, os._exit, 0)
    return web.json_response({"ok": True})


async def etat_piscine(request):
    return web.json_response(piscines.etat())


async def choisir_piscine(request):
    try:
        piscines.choisir((await request.json()).get("choix", ""))
    except ValueError as e:
        return web.json_response({"erreur": str(e)}, status=400)
    log(f"Piscine du client : {piscines.choix()}")
    return web.json_response(piscines.etat())


async def dicter_piscine(request):
    """Description de la piscine : enregistrement vocal (transcrit par Whisper) et/ou texte libre."""
    donnees = await request.post()
    morceaux = []
    voix = donnees.get("voix")
    if hasattr(voix, "file"):
        def transcrire(octets):
            with tempfile.TemporaryDirectory() as d:
                brut, wav = Path(d) / "brut.bin", Path(d) / "voix16k.wav"
                brut.write_bytes(octets)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(brut), "-ac", "1", "-ar", "16000",
                                str(wav)], check=True)
                segments, _ = WHISPER.transcribe(str(wav), language="fr", beam_size=5, vad_filter=True)
                return " ".join(s.text.strip() for s in segments).strip()
        try:
            dit = await asyncio.get_running_loop().run_in_executor(EXEC, transcrire, voix.file.read())
        except (OSError, subprocess.CalledProcessError) as e:
            return web.json_response({"erreur": f"enregistrement illisible ({e})"}, status=400)
        if dit:
            piscines.ajouter_dictee(dit, "Dicté")
            morceaux.append(dit)
    ecrit = str(donnees.get("texte", "")).strip()
    if ecrit:
        piscines.ajouter_dictee(ecrit, "Écrit")
        morceaux.append(ecrit)
    if not morceaux:
        return web.json_response({"erreur": "Rien d'audible ni d'écrit."}, status=400)
    log(f"Piscine dictée : {sum(len(m) for m in morceaux)} caractères ajoutés")
    return web.json_response({**piscines.etat(), "ajoute": " ".join(morceaux)})


async def effacer_piscine(request):
    piscines.effacer_dictee()
    log("Description dictée de la piscine effacée (demande depuis la page)")
    return web.json_response(piscines.etat())


async def version(request):
    """Ce qui tourne vraiment (menu ☰ → Version) : code, personnage, voix de référence, contrôle de la voix."""
    return web.json_response({**VERSION, "personnage": PERSONA, "voix_reference": Path(REF_AUDIO).name,
                              "hauteur_reference": round(REF_HAUTEUR or 0),
                              "plage": [round(b * (REF_HAUTEUR or 0)) for b in CONTROLE_BORNES],
                              "temperature": VOIX_TEMPERATURE, "modele_voix": TAILLE_VOIX, "mode_voix": VOIX_MODE, "vitesse": VOIX_VITESSE, "vitesse_attente": VITESSE_CLIPS,
                              "controle": "empreinte" if E_REF is not None else "hauteur",
                              "seuil_empreinte": SEUIL_EMPREINTE, "predictor": PREDICTOR or "0.9,50 (bibliothèque)",
                              **VOIX_STATS})


async def regler_vitesse(request):
    """Menu ☰ → Réglages : débit de la voix, appliqué dès la phrase suivante (pas de redémarrage).
    Les phrases d'attente gardent la vitesse du démarrage jusqu'au prochain redémarrage."""
    global VOIX_VITESSE
    try:
        v = round(float((await request.json())["vitesse"]), 2)
    except (ValueError, KeyError, TypeError):
        return web.json_response({"erreur": "vitesse invalide"}, status=400)
    if not 0.8 <= v <= 1.5:
        return web.json_response({"erreur": "vitesse entre 0,8 et 1,5"}, status=400)
    VOIX_VITESSE = v
    FICHIER_VITESSE.parent.mkdir(exist_ok=True)
    FICHIER_VITESSE.write_text(str(v), encoding="utf-8")
    log(f"Vitesse de la voix réglée à ×{v} (menu)")
    return web.json_response({"vitesse": v, "vitesse_attente": VITESSE_CLIPS})


async def page(request):
    return web.FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


async def photo(request):
    return web.FileResponse(SOURCE)


FICHIER_ACTIVITE = RACINE / "labo" / "journaux" / "activite.json"


def noter_activite():
    """Dernière utilisation (connexion, question) : lue par runpod/a_la_demande.py pour arrêter un pod inutilisé."""
    try:
        FICHIER_ACTIVITE.write_text(json.dumps({"derniere": time.time(), "connectes": len(SESSIONS)}), encoding="utf-8")
    except OSError:
        pass


async def websocket(request):
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
    await ws.prepare(request)
    loop = asyncio.get_running_loop()
    s = Session(ws, loop)
    VISAGE.sortie = s  # la dernière page connectée reçoit le visage
    tache = asyncio.create_task(s.expediteur())
    log("Navigateur connecté :", request.headers.get("Tailscale-User-Login") or request.remote)
    SESSIONS.add(s)
    noter_activite()
    for n in NOTES:
        s.envoyer(n)
    s.envoyer({"type": "etat", "etat": "ecoute"})
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            s.micro(np.frombuffer(msg.data, "<i2").astype(np.float32) / 32768)
        elif msg.type == WSMsgType.TEXT:
            d = json.loads(msg.data)
            if d["type"] == "lecture":
                m = s.mesures.get(d.get("rid"))
                if m is not None:
                    m.setdefault("t_lecture_attente" if d.get("attente") else "t_lecture", time.perf_counter())
                s.envoyer({"type": "etat", "etat": "parle"})
            elif d["type"] == "pret":
                s.cloturer(d.get("rid"), d.get("trous"))
                s.reprendre()
                s.t_pret = time.perf_counter()
                s.sourd_jusqua = s.t_pret + GARDE_ECHO
                s.envoyer({"type": "etat", "etat": "ecoute"})
            elif d["type"] == "texte" and not s.occupe and d.get("texte", "").strip():
                s.occupe = True
                s.envoyer({"type": "etat", "etat": "reflexion"})
                loop.run_in_executor(EXEC, s.traiter, None, time.perf_counter(), d["texte"].strip())
    SESSIONS.discard(s)
    s.file.put_nowait(None)
    await tache
    if VISAGE.sortie is s:
        VISAGE.sortie = None
    noter_activite()
    log("Navigateur déconnecté")
    return ws


def rendre_clip(phrase):
    """Voix clonée + visage pour une phrase d'attente → liste de paquets (pcm 40 ms, jpeg)."""
    audio = np.concatenate(EXEC.submit(lambda: list(parler_controle(phrase))).result())
    # silences de début et de fin coupés (phrase d'attente : 1 à 1,5 s, pas plus)
    fort = np.flatnonzero(np.abs(audio) > 0.02 * np.abs(audio).max())
    if len(fort):
        audio = audio[max(0, fort[0] - 1200):fort[-1] + 3600]
    class Capture:
        def __init__(self):
            self.paquets, self.fini = [], threading.Event()

        def image(self, rid, pcm, jpeg):
            self.paquets.append((pcm, jpeg))

        def fin(self, rid):
            self.fini.set()

    cap, ancienne = Capture(), VISAGE.sortie
    VISAGE.sortie = cap
    rid = f"attente:{phrase}"
    VISAGE.commencer(rid)
    VISAGE.pousser(audio)
    VISAGE.terminer()
    cap.fini.wait(60)
    VISAGE.sortie = ancienne
    VISAGE.t_premiere.pop(rid, None)
    VISAGE.t_derniere.pop(rid, None)
    VISAGE.n_images.pop(rid, None)
    return cap.paquets


# ─── Démarrage ───────────────────────────────────────────────────────────────
def charger():
    global WHISPER, TTS, VISAGE, CLIENT, DOCUMENTS
    t = time.perf_counter()
    WHISPER = WhisperModel("large-v3-turbo", device="cuda", device_index=GPU_VOIX, compute_type="float16",
                           download_root=str(RACINE / "models" / "whisper"))
    list(WHISPER.transcribe(np.zeros(16000, np.float32), language="fr")[0])
    log(f"Reconnaissance vocale prête ({time.perf_counter() - t:.0f} s)")

    t = time.perf_counter()
    TTS = FasterQwen3TTS.from_pretrained(str(RACINE / "models" / f"qwen3-tts-{TAILLE_VOIX}"),
                                         device=f"cuda:{GPU_VOIX}")
    # graphes CUDA capturés dans le fil qui servira ensuite
    EXEC.submit(lambda: list(TTS.generate_voice_clone_streaming(
        text="Bonjour.", language="French", ref_audio=REF_AUDIO, ref_text=REF_TEXTE,
        chunk_size=TTS_CHUNK, xvec_only=VOIX_XVEC))).result()
    global REF_HAUTEUR
    ref, sr_ref = sf.read(REF_AUDIO, dtype="float32")
    REF_HAUTEUR = hauteur_mediane(ref if sr_ref == 24000 else resample_poly(ref, 24000, sr_ref).astype(np.float32))
    log(f"Voix {TAILLE_VOIX} prête ({time.perf_counter() - t:.0f} s) — hauteur de la référence {REF_HAUTEUR:.0f} Hz, "
        f"plage acceptée {CONTROLE_BORNES[0] * REF_HAUTEUR:.0f}–{CONTROLE_BORNES[1] * REF_HAUTEUR:.0f} Hz")
    if PREDICTOR:  # hasard réduit du timbre : graphe du code predictor recapturé dans le fil qui servira (en mémoire)
        def recapturer():
            pg = TTS.predictor_graph
            pg.temperature, pg.top_k = (float(v) for v in PREDICTOR.split(","))
            pg.top_k = int(pg.top_k)
            pg.capture(num_warmup=3)
            list(TTS.generate_voice_clone_streaming(text="Bonjour.", language="French", ref_audio=REF_AUDIO,
                                                    ref_text=REF_TEXTE, chunk_size=TTS_CHUNK, xvec_only=VOIX_XVEC))
        EXEC.submit(recapturer).result()
        log(f"Code predictor recapturé : température {PREDICTOR.split(',')[0]}, top_k {PREDICTOR.split(',')[1]}")
    global E_REF
    if SEUIL_EMPREINTE is not None and PERSONA == "pierre":
        E_REF = EXEC.submit(lambda: empreinte_voix(
            ref if sr_ref == 24000 else resample_poly(ref, 24000, sr_ref).astype(np.float32))).result()
        log(f"Contrôle de la voix par empreinte : seuil {SEUIL_EMPREINTE:.2f}, jusqu'à {CONTROLE_V2_ESSAIS} essais "
            f"(phrases ≤ {CONTROLE_V2_MAX_CAR} caractères ; au-delà, contrôle par hauteur)")
    else:
        log("Contrôle de la voix par hauteur (v1) : " + ("seuil d'empreinte non fixé (AQUAPHONE_SEUIL_EMPREINTE)"
                                                        if SEUIL_EMPREINTE is None else f"personnage {PERSONA}, seuil calé sur Pierre"))

    t = time.perf_counter()
    if not SOURCE.exists():
        img = cv2.imread(str(RACINE / "assets" / "visage_pierre.jpg"))
        h, w = img.shape[:2]
        cv2.imwrite(str(SOURCE), cv2.resize(img, (round(w * HAUTEUR_SOURCE / h), HAUTEUR_SOURCE),
                                           interpolation=cv2.INTER_AREA))
    sdk = StreamSDK(str(RACINE / "models/ditto/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"),
                    str(RACINE / "models/ditto/ditto_pytorch"))
    # torch.compile dans le fil principal (dans les fils de Ditto, la compilation se bloque).
    # Le nombre de fils CPU de torch fait partie des gardes de compilation : on aligne le fil
    # principal sur la valeur d'un fil neuf (celle des fils de Ditto), sinon tout recompile.
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
    sortie = RACINE / "models" / "sorties_visage"
    sortie.mkdir(parents=True, exist_ok=True)
    sdk.setup(str(SOURCE), str(sortie / "serveur.mp4"), online_mode=True, overlap_v2=70, sampling_timesteps=PAS_DIFFUSION)
    try:
        sdk.writer.close() if hasattr(sdk.writer, "close") else None
    except Exception:
        pass
    VISAGE = Visage(sdk)
    # préchauffage : 2 s de silence
    VISAGE.commencer(None)
    VISAGE.pousser(np.zeros(2 * 24000, np.float32))
    VISAGE.terminer()
    limite = time.perf_counter() + 60
    while ((VISAGE.n_nourries < 50 or VISAGE.n_emises < VISAGE.n_nourries - 25)
           and time.perf_counter() < limite):
        time.sleep(0.1)
    time.sleep(0.5)
    log(f"Visage prêt ({time.perf_counter() - t:.0f} s) — images données {VISAGE.n_nourries}, "
        f"produites {VISAGE.n_emises}")

    t = time.perf_counter()
    global CLIPS
    CLIPS = attente.charger(rendre_clip, ("silences coupés", TAILLE_VOIX, PAS_DIFFUSION, HAUTEUR_SOURCE, LARGEUR_VIDEO, QUALITE_JPEG,
                                          os.path.getsize(REF_AUDIO), REF_TEXTE, PERSONA, VOIX_TEMPERATURE,
                                          # v2 : autres tirages → clips à refaire (les anciens .pkl restent sur le disque)
                                          f"contrôle voix 2 {SEUIL_EMPREINTE} {PREDICTOR}" if E_REF is not None
                                          else "contrôle voix 1" + (f" {PREDICTOR}" if PREDICTOR else ""))
                                         + (("empreinte seule",) if VOIX_XVEC else ())  # nouveaux clips en mode B
                                         + ((f"vitesse {VITESSE_CLIPS}",) if VITESSE_CLIPS != 1.0 else ()))
    log(f"Phrases d'attente prêtes ({time.perf_counter() - t:.0f} s) : " + ", ".join(
        f"« {p} » {len(c) / IPS:.1f} s" for p, c in CLIPS.items()))

    CLIENT = anthropic.Anthropic()
    DOCUMENTS = contexte.documents()
    if DOCUMENTS:
        # écrit la documentation dans le cache de l'API (sinon la 1re question paierait cette écriture)
        t = time.perf_counter()
        mc = {}
        list(cerveau.phrases_en_flux(CLIENT, [{"role": "user", "content": "Bonjour."}], mesures=mc,
                                     documents=DOCUMENTS))
        u = mc.get("usage")
        log(f"Contexte iopool : {len(DOCUMENTS)} caractères, cache écrit {u.cache_creation_input_tokens} / "
            f"lu {u.cache_read_input_tokens} tokens ({time.perf_counter() - t:.1f} s)")
    else:
        log("Contexte iopool absent (dossier connaissances/) : prompt de base seul")
    for i in range(torch.cuda.device_count()):
        log(f"Carte {i} ({torch.cuda.get_device_name(i)}) : {torch.cuda.memory_allocated(i) / 1e9:.1f} Go utilisés"
            f"{' — voix + transcription' if i == GPU_VOIX else ''}{' — visage' if i == 0 else ''}")


def main():
    charger()
    if HOTE != "127.0.0.1" and not acces.actif():
        sys.exit("Refus : AQUAPHONE_HOTE public sans AQUAPHONE_MOT_DE_PASSE dans .env")

    async def connexion(request):
        return await acces.connexion(request, log)

    app = web.Application(middlewares=[acces.garde], client_max_size=40 * 1024 * 1024)
    app.add_routes([web.get("/", page), web.get("/photo.jpg", photo), web.get("/ws", websocket),
                    web.post("/note", note), web.get("/connexion", connexion), web.post("/connexion", connexion),
                    web.get("/personas", liste_personas), web.get("/personas/{nom}/photo", photo_persona),
                    web.post("/personas/{nom}", deposer_persona), web.post("/persona", choisir_persona),
                    web.get("/version", version), web.post("/voix/vitesse", regler_vitesse),web.get("/piscine", etat_piscine), web.post("/piscine/choix", choisir_piscine),
                    web.post("/piscine/dictee", dicter_piscine), web.post("/piscine/effacer", effacer_piscine)])
    FICHIER_JETON.parent.mkdir(exist_ok=True)
    FICHIER_JETON.write_text(JETON, encoding="utf-8")
    log(f"Serveur prêt : http://{HOTE}:{PORT}/  (via Tailscale : tailscale serve --bg {PORT}) — "
        f"mot de passe {'actif' if acces.actif() else 'inactif'}")
    web.run_app(app, host=HOTE, port=PORT, print=None)


if __name__ == "__main__":
    main()
