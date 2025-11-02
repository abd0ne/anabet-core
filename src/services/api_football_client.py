import httpx
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
from src.configs.api_football_config import get_api_football_settings
from src.services.cache_service import cache_service
from src.services.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class APIFootballException(Exception):
    """Exception personnalisée pour les erreurs API Football"""
    pass

class APIFootballClient:
    """Client pour interagir avec l'API Football"""
    
    def __init__(self):
        self.settings = get_api_football_settings()
        self.base_url = self.settings.api_football_base_url
        self.headers = {
            'x-rapidapi-key': self.settings.api_football_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        self.rate_limiter = RateLimiter(
            max_requests=self.settings.rate_limit_per_minute,
            time_window=60
        )
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Context manager entry"""
        self.client = httpx.AsyncClient(
            timeout=self.settings.api_football_timeout,
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.client:
            await self.client.aclose()
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Effectue une requête à l'API avec retry logic et rate limiting
        
        Args:
            endpoint: L'endpoint de l'API (ex: '/fixtures')
            params: Paramètres de la requête
            use_cache: Utiliser le cache ou non
        
        Returns:
            Réponse JSON de l'API
        """
        if params is None:
            params = {}
        
        # Vérifier le cache
        if use_cache:
            cached_data = cache_service.get(endpoint, params)
            if cached_data is not None:
                logger.info(f"Cache hit for {endpoint} with params {params}")
                return cached_data
        
        # Rate limiting
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.get_wait_time()
            if wait_time and wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
        
        # Retry logic
        last_exception = None
        for attempt in range(self.settings.max_retries):
            try:
                self.rate_limiter.add_request()
                
                url = f"{self.base_url}{endpoint}"
                logger.info(f"Making request to {url} with params {params} (attempt {attempt + 1})")
                
                if not self.client:
                    self.client = httpx.AsyncClient(
                        timeout=self.settings.api_football_timeout,
                        headers=self.headers
                    )
                
                response = await self.client.get(url, params=params)
                
                # Vérifier le statut
                if response.status_code == 200:
                    data = response.json()
                    
                    # Vérifier si l'API a retourné une erreur
                    if 'errors' in data and data['errors']:
                        raise APIFootballException(f"API Error: {data['errors']}")
                    
                    # Mettre en cache
                    if use_cache:
                        cache_service.set(endpoint, params, data, self.settings.cache_ttl)
                    
                    return data
                
                elif response.status_code == 429:
                    # Too many requests
                    logger.warning("Rate limit exceeded (429). Waiting before retry...")
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                
                elif response.status_code == 499:
                    # Quota exceeded
                    raise APIFootballException("API quota exceeded. Please upgrade your plan.")
                
                else:
                    raise APIFootballException(
                        f"HTTP {response.status_code}: {response.text}"
                    )
            
            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2 * (attempt + 1))
            
            except httpx.RequestError as e:
                last_exception = e
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2 * (attempt + 1))
        
        # Si tous les essais ont échoué
        raise APIFootballException(
            f"Failed after {self.settings.max_retries} attempts. Last error: {last_exception}"
        )
    
    # ==================== LEAGUES ====================
    
    async def get_leagues(
        self,
        country: Optional[str] = None,
        season: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère la liste des ligues
        
        Args:
            country: Filtrer par pays (ex: 'France', 'England')
            season: Filtrer par saison (ex: 2024)
        """
        params = {}
        if country:
            params['country'] = country
        if season:
            params['season'] = season
        
        response = await self._make_request('/leagues', params)
        return response.get('response', [])
    
    # ==================== TEAMS ====================
    
    async def get_team(self, team_id: int) -> Optional[Dict[str, Any]]:
        """Récupère les informations d'une équipe"""
        response = await self._make_request('/teams', {'id': team_id})
        teams = response.get('response', [])
        return teams[0] if teams else None
    
    async def search_teams(
        self,
        name: Optional[str] = None,
        country: Optional[str] = None,
        league: Optional[int] = None,
        season: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche des équipes
        
        Args:
            name: Nom de l'équipe
            country: Pays
            league: ID de la ligue
            season: Saison
        """
        params = {}
        if name:
            params['search'] = name
        if country:
            params['country'] = country
        if league:
            params['league'] = league
        if season:
            params['season'] = season
        
        response = await self._make_request('/teams', params)
        return response.get('response', [])
    
    async def get_team_statistics(
        self,
        team_id: int,
        league_id: int,
        season: int
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère les statistiques d'une équipe pour une saison
        
        Args:
            team_id: ID de l'équipe
            league_id: ID de la ligue
            season: Année de la saison
        """
        params = {
            'team': team_id,
            'league': league_id,
            'season': season
        }
        
        response = await self._make_request('/teams/statistics', params)
        return response.get('response', None)
    
    # ==================== FIXTURES (MATCHES) ====================
    
    async def get_fixtures(
        self,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        team_id: Optional[int] = None,
        date: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        last: Optional[int] = None,
        next: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère les matchs selon différents critères
        
        Args:
            league_id: ID de la ligue
            season: Saison
            team_id: ID de l'équipe
            date: Date spécifique (YYYY-MM-DD)
            from_date: Date de début (YYYY-MM-DD)
            to_date: Date de fin (YYYY-MM-DD)
            last: N derniers matchs d'une équipe
            next: N prochains matchs d'une équipe
            status: Statut du match (NS, LIVE, FT, etc.)
        """
        params = {}
        if league_id:
            params['league'] = league_id
        if season:
            params['season'] = season
        if team_id:
            params['team'] = team_id
        if date:
            params['date'] = date
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        if last:
            params['last'] = last
        if next:
            params['next'] = next
        if status:
            params['status'] = status
        
        response = await self._make_request('/fixtures', params, use_cache=True)
        return response.get('response', [])
    
    async def get_fixture_by_id(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """Récupère un match par son ID"""
        response = await self._make_request('/fixtures', {'id': fixture_id})
        fixtures = response.get('response', [])
        return fixtures[0] if fixtures else None
    
    async def get_fixture_statistics(
        self,
        fixture_id: int
    ) -> List[Dict[str, Any]]:
        """
        Récupère les statistiques détaillées d'un match
        (tirs, possession, xG, etc.)
        """
        response = await self._make_request(
            '/fixtures/statistics',
            {'fixture': fixture_id}
        )
        return response.get('response', [])
    
    async def get_head_to_head(
        self,
        team1_id: int,
        team2_id: int,
        last: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des confrontations entre deux équipes
        
        Args:
            team1_id: ID de la première équipe
            team2_id: ID de la deuxième équipe
            last: Nombre de derniers matchs à récupérer
        """
        params = {'h2h': f"{team1_id}-{team2_id}"}
        if last:
            params['last'] = last
        
        response = await self._make_request('/fixtures/headtohead', params)
        return response.get('response', [])
    
    # ==================== STANDINGS ====================
    
    async def get_standings(
        self,
        league_id: int,
        season: int,
        team_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère le classement d'une ligue
        
        Args:
            league_id: ID de la ligue
            season: Saison
            team_id: Filtrer par équipe (optionnel)
        """
        params = {
            'league': league_id,
            'season': season
        }
        if team_id:
            params['team'] = team_id
        
        response = await self._make_request('/standings', params)
        return response.get('response', [])
    
    # ==================== PLAYERS ====================
    
    async def get_players(
        self,
        team_id: Optional[int] = None,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        player_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère les informations des joueurs
        
        Args:
            team_id: ID de l'équipe
            league_id: ID de la ligue
            season: Saison
            player_id: ID du joueur
        """
        params = {}
        if team_id:
            params['team'] = team_id
        if league_id:
            params['league'] = league_id
        if season:
            params['season'] = season
        if player_id:
            params['id'] = player_id
        
        response = await self._make_request('/players', params)
        return response.get('response', [])
    
    async def get_top_scorers(
        self,
        league_id: int,
        season: int
    ) -> List[Dict[str, Any]]:
        """Récupère le classement des buteurs"""
        params = {
            'league': league_id,
            'season': season
        }
        
        response = await self._make_request('/players/topscorers', params)
        return response.get('response', [])
    
    async def get_top_assists(
        self,
        league_id: int,
        season: int
    ) -> List[Dict[str, Any]]:
        """Récupère le classement des passeurs"""
        params = {
            'league': league_id,
            'season': season
        }
        
        response = await self._make_request('/players/topassists', params)
        return response.get('response', [])
    
    # ==================== ODDS ====================
    
    async def get_odds(
        self,
        fixture_id: Optional[int] = None,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        date: Optional[str] = None,
        bookmaker: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère les cotes des bookmakers
        
        Args:
            fixture_id: ID du match
            league_id: ID de la ligue
            season: Saison
            date: Date (YYYY-MM-DD)
            bookmaker: ID du bookmaker
        """
        params = {}
        if fixture_id:
            params['fixture'] = fixture_id
        if league_id:
            params['league'] = league_id
        if season:
            params['season'] = season
        if date:
            params['date'] = date
        if bookmaker:
            params['bookmaker'] = bookmaker
        
        response = await self._make_request('/odds', params)
        return response.get('response', [])
    
    # ==================== PREDICTIONS ====================
    
    async def get_predictions(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """
        Récupère les prédictions de l'API pour un match
        (forme, comparaison, prédiction de résultat)
        """
        response = await self._make_request('/predictions', {'fixture': fixture_id})
        predictions = response.get('response', [])
        return predictions[0] if predictions else None
    
    # ==================== UTILITY METHODS ====================
    
    async def get_last_n_matches_home(
        self,
        team_id: int,
        league_id: int,
        season: int,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Récupère les N derniers matchs à domicile d'une équipe"""
        fixtures = await self.get_fixtures(
            team_id=team_id,
            league_id=league_id,
            season=season,
            last=50  # Récupérer plus pour filtrer ensuite
        )
        
        # Filtrer les matchs à domicile
        home_fixtures = [
            f for f in fixtures
            if f['teams']['home']['id'] == team_id and f['fixture']['status']['short'] == 'FT'
        ]
        
        # Trier par date décroissante et prendre les N premiers
        home_fixtures.sort(key=lambda x: x['fixture']['date'], reverse=True)
        return home_fixtures[:n]
    
    async def get_last_n_matches_away(
        self,
        team_id: int,
        league_id: int,
        season: int,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Récupère les N derniers matchs à l'extérieur d'une équipe"""
        fixtures = await self.get_fixtures(
            team_id=team_id,
            league_id=league_id,
            season=season,
            last=50
        )
        
        # Filtrer les matchs à l'extérieur
        away_fixtures = [
            f for f in fixtures
            if f['teams']['away']['id'] == team_id and f['fixture']['status']['short'] == 'FT'
        ]
        
        # Trier par date décroissante et prendre les N premiers
        away_fixtures.sort(key=lambda x: x['fixture']['date'], reverse=True)
        return away_fixtures[:n]
    
    def get_rate_limiter_stats(self) -> dict:
        """Retourne les statistiques du rate limiter"""
        return self.rate_limiter.get_stats()
    
    def get_cache_stats(self) -> dict:
        """Retourne les statistiques du cache"""
        return cache_service.get_stats()

# Instance globale
api_football_client = APIFootballClient()