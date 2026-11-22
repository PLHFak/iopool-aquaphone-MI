#!/usr/bin/env bash
# Aquaphone-MI : relance le serveur dès qu'il s'arrête (changement de personnage depuis le menu, ou plantage).
cd "$HOME"/iopool-aquaphone
while true; do
  .venv/bin/python -u labo/serveur.py >> labo/journaux/serveur.log 2>&1
  echo "$(date +%H:%M:%S) serveur arrêté (code $?) — relance" >> labo/journaux/serveur.log
  sleep 2
done
