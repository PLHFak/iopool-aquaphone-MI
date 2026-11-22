#!/usr/bin/env bash
# Aquaphone-MI : démarre le serveur Aquaphone en local dans WSL2
# (une seule RTX 4090 : visage et voix se partagent la carte).
# Accès depuis Windows : http://localhost:8765 (redirection automatique WSL2 → navigateur Windows).
# Le serveur refuse de démarrer sans mot de passe ; rien n'est exposé sur Internet.
set -euo pipefail
D=$HOME/iopool-aquaphone
cd "$D"
grep -q '^AQUAPHONE_MOT_DE_PASSE=.' .env || { echo "Manque AQUAPHONE_MOT_DE_PASSE dans .env"; exit 1; }
mkdir -p labo/journaux
export AQUAPHONE_GPU_VOIX=0 AQUAPHONE_HOTE=127.0.0.1 PYTHONIOENCODING=utf-8
pkill -f windows/boucle.sh 2>/dev/null || true
pkill -f labo/serveur.py 2>/dev/null || true
: > labo/journaux/serveur.log
nohup bash windows/boucle.sh > /dev/null 2>&1 &
echo "Aquaphone-MI lancé (journal : tail -f labo/journaux/serveur.log, prêt en ≈ 1 à 2 min)."
echo "Adresse : http://localhost:8765/  (depuis le navigateur Windows)"
