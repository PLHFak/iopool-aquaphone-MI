# Aquaphone-MI — WSL2 sur le PC Windows (RTX 4090)

**Aquaphone-MI** (MI = **MI**stral) est la variante **locale** d'Aquaphone : le code de synthèse
vocale et d'animation du visage du dépôt [PLHFak/IOPOOL-aquaphone](https://github.com/PLHFak/IOPOOL-aquaphone),
exécuté sur votre PC Windows via WSL2, sans dépendre du pod. Le cerveau reste **Claude (API)** ;
les outils de test (`labo2/clips_local.py`) n'en ont pas besoin du tout.

| | Aquaphone (production) | Aquaphone-MI (ce dépôt) |
|---|---|---|
| Où | Pod RunPod, 2 × RTX 4090 | Votre PC Windows, 1 × RTX 4090 dans WSL2 |
| Service | Complet : accueil, micro, Claude, escalade | Serveur complet possible, ou clips locaux hors LLM |
| Accès | Adresse publique protégée par mot de passe | Local (127.0.0.1) ; exposition temporaire possible via tunnel (ngrok), mot de passe du serveur requis |
| Usage | Production et testeurs | Débogage, mesures, calage du seuil de voix, coût nul |

## Contenu du kit

- `labo/serveur.py`, `labo/web/index.html` : **surcouche MI** posée sur le dépôt de base par
  `windows/installer.sh` (le dépôt de base n'est jamais modifié sur GitHub).
- `labo2/clips_local.py` : clip local SANS LLM — texte → voix clonée (Qwen3-TTS) → visage animé (Ditto) → MP4.
- `windows/installer.py` (+ `installer.sh`) : installation complète dans WSL2 (idempotente).
- `windows/lancer.sh` + `windows/boucle.sh` : démarrage du serveur local et relance automatique.

## Modifications MI (22/11/2026) par rapport au dépôt de base

1. **Vidéo HD** : largeur envoyée au navigateur configurable par `AQUAPHONE_LARGEUR` (720 posé par
   l'installateur MI, au lieu de 480). Le cache des phrases d'attente suit automatiquement.
2. **Compression adaptative (« VBR »)** : chaque image JPEG part à la qualité cible `AQUAPHONE_JPEG`
   (84) ; si elle dépasse `AQUAPHONE_BUDGET_IMAGE` (40 000 octets), sa qualité est réduite par crans
   jusqu'au plancher 55. Le visage bouge peu : la plupart des images restent en qualité maximale et
   le flux reste léger pour internet.
3. **Menu ☰ explicite** : chaque réglage de la page (version en service, personnage, piscine du
   client, vitesse du phrasé, mesures de délai) porte son explication directe, pensée pour les
   clients et les testeurs qui se posent des questions sur les paramètres.

## Installation (une seule fois, ≈ 30 min + téléchargements)

```bash
sudo apt-get update && sudo apt-get install -y git python3-venv
git clone https://github.com/PLHFak/iopool-aquaphone-MI.git ~/iopool-aquaphone-MI
bash ~/iopool-aquaphone-MI/windows/installer.sh
```

L'installateur clones le dépôt de base dans `~/iopool-aquaphone`, pose la surcouche MI, installe le
Python, Ditto (avec le correctif de recollage) et les modèles (~8 Go). Prérequis Windows : pilote
NVIDIA à jour, WSL2 (`wsl --install -d Ubuntu-24.04`), la RTX 4090 visible via `nvidia-smi`.

## Fichiers privés (copiés à la main, jamais via Git)

```bash
cd ~/iopool-aquaphone
cp /mnt/c/Users/<vous>/…/.env .                       # ANTHROPIC_API_KEY + AQUAPHONE_MOT_DE_PASSE
cp /mnt/c/…/visage_pierre.jpg assets/
cp /mnt/c/…/ref_pierre.wav models/                    # mono 24 kHz
cp /mnt/c/…/iopool-agent-reference-docs connaissances/
```

## Lancer

```bash
bash ~/iopool-aquaphone-MI/windows/lancer.sh     # serveur complet (Claude inclus)
```
Puis dans le navigateur Windows : `http://localhost:8765/` (mot de passe = `AQUAPHONE_MOT_DE_PASSE`).
Clips sans LLM : `cd ~/iopool-aquaphone && .venv/bin/python labo2/clips_local.py "Bonjour."`

## Réglages locaux (une seule carte)

- `AQUAPHONE_GPU_VOIX=0` (fixé par `lancer.sh`) : tout sur la carte 0 ; si le processeur âgé prend
  du retard sur le visage : réduire `AQUAPHONE_LARGEUR` (540) ou `AQUAPHONE_BUDGET_IMAGE` (30000).
- `AQUAPHONE_SEUIL_EMPREINTE` / `AQUAPHONE_PREDICTOR` : pas encore de valeurs mesurées —
  voir `labo2/BRIEF_CORRECTION_VOIX.md` et `labo2/essai_empreinte.py`.

## Confidentialité

- Le serveur écoute sur `127.0.0.1` : accessible depuis ce PC. Toute exposition (ngrok, Tailscale)
  reste protégée par le mot de passe du serveur ; photo, voix et `.env` restent sur la machine.
