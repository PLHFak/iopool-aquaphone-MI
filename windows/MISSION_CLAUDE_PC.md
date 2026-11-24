# Mission de Claude Code sur le PC labo — mettre Aquaphone-MI en service

Tu tournes sur le PC Windows de Pierre (RTX 4090, 24 Go). Lis d'abord `README.md` de ce dépôt.

## Le projet

Aquaphone-MI est le projet que Pierre veut faire vivre : un pack de **test interne** (pas un produit
commercial) ouvert à quelques personnes choisies par Pierre. Le PC lui est dédié, allumé 24 h/24.
Cerveau : Claude (API), avec la documentation iopool et la piscine du client. Voix (Qwen3-TTS) et
visage (Ditto) tournent sur la 4090, dans WSL2.

Répartition du travail :
- **Toi (PC)** : installer, lancer, faire démarrer tout seul, ouvrir l'accès. Tu ne modifies pas le code
  du serveur (`labo/serveur.py`, `labo/web/`).
- **Claude Code sur le Mac mini** : développe le code (accueil de plusieurs personnes avec réduction du
  service par paliers, codes d'accès par testeur) et le publie sur GitHub. Tu récupères ces mises à jour
  par `git pull` quand Pierre te le demande.

## Règles (de Pierre, à respecter)

- **Pierre ne tape aucune commande** : tu exécutes tout toi-même. Ne lui demande une action que si elle
  est impossible depuis le terminal (clic dans une fenêtre, mot de passe), en lui disant exactement où cliquer.
- Réponds en français, court et structuré. Pierre dicte à la voix : interprète les erreurs de transcription.
- Annonce ce que tu trouves avant d'installer. Annonce la taille avant tout téléchargement de plusieurs Go.
- Ne modifie ni les pilotes NVIDIA ni les logiciels Windows existants sans demander.
- N'arrête aucun service existant (autre serveur sur le port 8765, DUPLEX, Ollama…) sans demander.
- Fichiers privés (`.env`, `assets/`, `models/`, `connaissances/`) : jamais dans Git, le dépôt est public.
- Visage et voix : uniquement ceux de Pierre.
- Ne lance pas de test automatique pendant que quelqu'un utilise la page.

## Étapes

1. **État des lieux** (sans rien installer) : WSL2 et sa distribution, `nvidia-smi` dans WSL2, espace
   disque, présence de `~/iopool-aquaphone-MI` et `~/iopool-aquaphone` dans WSL2, ce qui écoute déjà sur
   le port 8765 (Windows et WSL2), état de Tailscale sur Windows. Fais le rapport à Pierre.
2. **Installation** si elle manque ou est incomplète : `windows/installer.sh` (≈ 8 Go de modèles, annonce-le).
3. **Fichiers privés** dans `~/iopool-aquaphone` (WSL2) : reprends ceux qui existent déjà sur le PC
   (ancien dépôt Aquaphone côté Windows : `.env`, `assets/`, `models/ref_pierre.wav`, `connaissances/`).
   Sinon, ils sont dans le Google Drive de Pierre, dossier
   `IoT Faktory PLH/ioPool/IOPOOL AQUAPHONE VIDEO IMAGE`. Le `.env` doit contenir `ANTHROPIC_API_KEY` et
   `AQUAPHONE_MOT_DE_PASSE` : s'il manque l'un des deux, demande-le à Pierre.
4. **Lancement** : `windows/lancer.sh`, puis vérifie que la page répond sur `http://localhost:8765/`
   et fais un essai réel avec Pierre (une phrase, voix et visage). Note les délais vus dans le journal.
5. **Démarrage automatique** : à l'ouverture de session Windows (tâche planifiée qui lance WSL2 et
   `windows/lancer.sh`), et relance en cas de plantage (`windows/boucle.sh` le fait déjà). Vérifie que
   Windows ne met pas le PC en veille.
6. **Accès de l'extérieur** : propose à Pierre une adresse HTTPS permanente (Tailscale Funnel sur le port
   8765 de préférence, protégée par le mot de passe du serveur). **Demande son accord avant de l'ouvrir
   sur Internet.**
7. **Rapport final** à Pierre : adresse, mot de passe où il est rangé, délais mesurés, ce qui reste ouvert.
   Écris-le aussi dans `~/iopool-aquaphone/labo/resultats_mi.md` (non commité).
