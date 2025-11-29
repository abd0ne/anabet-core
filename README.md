# Anabet Core

API d'analyse sportive avec IA et intégration API-Football.

## Description

Anabet Core est une API REST construite avec FastAPI qui fournit des analyses de matchs de football en utilisant l'intelligence artificielle (Ollama) et l'intégration avec l'API Football. Le système offre des prédictions, des statistiques et des analyses détaillées des matchs.

## Fonctionnalités

- 🧠 **Prédictions IA** : Analyse de matchs avec Ollama (modèle gpt-oss:20b)
- ⚽ **Intégration API-Football** : Accès aux données de football en temps réel
- 🔍 **Recherche web** : Intégration Tavily pour enrichir les analyses
- 💾 **Cache** : Système de cache pour optimiser les performances
- ⏱️ **Rate Limiting** : Limitation du taux de requêtes pour protéger l'API
- 🏥 **Gestion des blessures** : Récupération et formatage des données de blessures

## Prérequis

- Python >= 3.12
- Ollama installé et configuré (http://localhost:11434)
- Clé API Tavily (TAVILY_API_KEY)
- Clé API Football (API_FOOTBALL_KEY)

## Installation

1. Cloner le repository :
```bash
git clone <repository-url>
cd anabet-core
```

2. Installer les dépendances avec uv :
```bash
uv sync
```

3. Configurer les variables d'environnement :
Créer un fichier `.env` avec :
```
TAVILY_API_KEY=your_tavily_api_key
API_FOOTBALL_KEY=your_api_football_key
API_FOOTBALL_BASE_URL=https://v3.football.api-sports.io
```

## Démarrage

Lancer le serveur :
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

Ou directement :
```bash
python main.py
```

L'API sera accessible sur `http://localhost:8001`

## Documentation API

Une fois le serveur démarré, la documentation interactive est disponible sur :
- Swagger UI : `http://localhost:8001/docs`
- ReDoc : `http://localhost:8001/redoc`

## Endpoints principaux

### Analyse de matchs

#### `POST /api/football/analyze-match`
Analyse un match avec l'IA en utilisant les noms des équipes et la date.

**Body:**
```json
{
  "team_a": "Paris Saint-Germain",
  "team_b": "Olympique de Marseille",
  "date": "2024-01-15"
}
```

**Réponse:**
```json
{
  "oneXTwo": {
    "equipe_a": 50.0,
    "equipe_x": 30.0,
    "equipe_b": 20.0,
    "justification": "..."
  },
  "xg_equipe_a": 1.5,
  "xg_equipe_b": 1.2,
  "tirs_attendus": {
    "equipe_a": 10,
    "equipe_b": 8
  },
  "probabilite_clean_sheet": {
    "equipe_a": 50.0,
    "equipe_b": 30.0,
    "justification": "..."
  },
  "pourcentages_plus_moins_2.5_buts": {
    "plus": 50.0,
    "moins": 30.0,
    "justification": "..."
  }
}
```

### Ligues

#### `GET /api/football/leagues`
Récupère la liste des ligues.

**Paramètres:**
- `country` (optionnel) : Nom du pays
- `season` (optionnel) : Année de la saison

**Exemple:**
```
GET /api/football/leagues?season=2023
```

### Équipes

#### `GET /api/football/teams/search`
Recherche des équipes.

**Paramètres:**
- `name` (optionnel) : Nom de l'équipe
- `country` (optionnel) : Pays
- `league` (optionnel) : ID de la ligue
- `season` (optionnel) : Saison

#### `GET /api/football/teams/{team_id}`
Récupère les informations d'une équipe.

### Matchs (Fixtures)

#### `GET /api/football/fixtures`
Récupère les matchs selon différents critères.

**Paramètres:**
- `league_id` (optionnel) : ID de la ligue
- `season` (optionnel) : Saison
- `team_id` (optionnel) : ID de l'équipe
- `date` (optionnel) : Date (YYYY-MM-DD)
- `from_date` (optionnel) : Date de début
- `to_date` (optionnel) : Date de fin
- `last` (optionnel) : N derniers matchs
- `next` (optionnel) : N prochains matchs

#### `GET /api/football/fixtures/{fixture_id}`
Récupère un match par son ID.

#### `GET /api/football/fixtures/{fixture_id}/statistics`
Récupère les statistiques d'un match.

### Confrontations

#### `GET /api/football/head-to-head`
Récupère l'historique des confrontations entre deux équipes.

**Paramètres:**
- `team1_id` : ID de l'équipe 1
- `team2_id` : ID de l'équipe 2
- `last` (optionnel) : Nombre de matchs

### Classements

#### `GET /api/football/standings`
Récupère le classement d'une ligue.

**Paramètres:**
- `league_id` : ID de la ligue
- `season` : Saison
- `team_id` (optionnel) : ID de l'équipe

### Joueurs

#### `GET /api/football/players/top-scorers`
Récupère le classement des buteurs.

**Paramètres:**
- `league_id` : ID de la ligue
- `season` : Saison

#### `GET /api/football/players/top-assists`
Récupère le classement des passeurs.

**Paramètres:**
- `league_id` : ID de la ligue
- `season` : Saison

### Blessures

#### `GET /api/football/injuries`
Récupère la liste des joueurs blessés.

**Paramètres:**
- `fixture_id` (optionnel) : ID du match
- `league_id` (optionnel) : ID de la ligue
- `season` (optionnel) : Saison
- `team_id` (optionnel) : ID de l'équipe
- `player_id` (optionnel) : ID du joueur
- `date` (optionnel) : Date (YYYY-MM-DD)
- `timezone` (optionnel) : Timezone

### Prédictions

#### `GET /api/football/predictions/{fixture_id}`
Récupère les prédictions de l'API pour un match.

### Statistiques

#### `GET /api/football/stats/rate-limiter`
Récupère les statistiques du rate limiter.

#### `GET /api/football/stats/cache`
Récupère les statistiques du cache.

## Structure du projet

```
anabet-core/
├── main.py                 # Point d'entrée de l'application
├── pyproject.toml          # Configuration du projet
├── README.md              # Documentation
└── src/
    ├── api/
    │   └── api_football_controller.py  # Contrôleurs API
    ├── configs/
    │   └── api_football_config.py      # Configuration
    └── services/
        ├── agent.py                     # Agent IA pour l'analyse
        ├── api_football_client.py       # Client API Football
        ├── cache_service.py             # Service de cache
        └── rate_limiter.py              # Rate limiter
```

## Technologies utilisées

- **FastAPI** : Framework web moderne et rapide
- **LangChain** : Framework pour applications LLM
- **LangChain Ollama** : Intégration avec Ollama
- **LangChain Tavily** : Recherche web
- **LangGraph** : Graphiques d'agents
- **Pydantic** : Validation de données
- **httpx** : Client HTTP asynchrone

## Configuration de l'agent IA

L'agent utilise :
- **Modèle** : gpt-oss:20b (via Ollama)
- **Temperature** : 0.3
- **Outil de recherche** : Tavily Search
- **Format de réponse** : JSON structuré avec Pydantic

## Gestion des erreurs

L'API gère automatiquement :
- Les timeouts
- Les erreurs de rate limiting
- Les erreurs de connexion
- La récupération automatique du client HTTP

## Développement

Pour contribuer au projet :

1. Créer une branche pour votre fonctionnalité
2. Faire vos modifications
3. Tester localement
4. Créer une pull request

## Licence

[À définir]

## Support

Pour toute question ou problème, ouvrir une issue sur le repository.

