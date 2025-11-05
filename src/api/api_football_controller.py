from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

from pydantic import BaseModel
from src.services.agent import analyze_match
from src.services.api_football_client import api_football_client, APIFootballException

class LLMRequest(BaseModel):
    team_a: str
    team_b: str

router = APIRouter(prefix="/api/football", tags=["api-football"])

@router.get("/leagues")
async def get_leagues(
    country: Optional[str] = Query(None, description="Nom du pays"),
    season: Optional[int] = Query(None, description="Année de la saison")
):
    """Récupère la liste des ligues"""
    try:
        async with api_football_client as client:
            leagues = await client.get_leagues(country=country, season=season)
            return {"success": True, "count": len(leagues), "data": leagues}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/teams/search")
async def search_teams(
    name: Optional[str] = Query(None, description="Nom de l'équipe"),
    country: Optional[str] = Query(None, description="Pays"),
    league: Optional[int] = Query(None, description="ID de la ligue"),
    season: Optional[int] = Query(None, description="Saison")
):
    """Recherche des équipes"""
    try:
        async with api_football_client as client:
            teams = await client.search_teams(
                name=name,
                country=country,
                league=league,
                season=season
            )
            return {"success": True, "count": len(teams), "data": teams}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/teams/{team_id}")
async def get_team(team_id: int):
    """Récupère les informations d'une équipe"""
    try:
        async with api_football_client as client:
            team = await client.get_team(team_id)
            if not team:
                raise HTTPException(status_code=404, detail="Team not found")
            return {"success": True, "data": team}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/teams/{team_id}/statistics")
async def get_team_statistics(
    team_id: int,
    league_id: int = Query(..., description="ID de la ligue"),
    season: int = Query(..., description="Saison")
):
    """Récupère les statistiques d'une équipe"""
    try:
        async with api_football_client as client:
            stats = await client.get_team_statistics(team_id, league_id, season)
            if not stats:
                raise HTTPException(status_code=404, detail="Statistics not found")
            return {"success": True, "data": stats}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fixtures")
async def get_fixtures(
    league_id: Optional[int] = Query(None, description="ID de la ligue"),
    season: Optional[int] = Query(None, description="Saison"),
    team_id: Optional[int] = Query(None, description="ID de l'équipe"),
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD)"),
    from_date: Optional[str] = Query(None, description="Date de début"),
    to_date: Optional[str] = Query(None, description="Date de fin"),
    last: Optional[int] = Query(None, description="N derniers matchs"),
    next: Optional[int] = Query(None, description="N prochains matchs")
):
    """Récupère les matchs"""
    try:
        async with api_football_client as client:
            fixtures = await client.get_fixtures(
                league_id=league_id,
                season=season,
                team_id=team_id,
                date=date,
                from_date=from_date,
                to_date=to_date,
                last=last,
                next=next
            )
            return {"success": True, "count": len(fixtures), "data": fixtures}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fixtures/{fixture_id}")
async def get_fixture(fixture_id: int):
    """Récupère un match par son ID"""
    try:
        async with api_football_client as client:
            fixture = await client.get_fixture_by_id(fixture_id)
            if not fixture:
                raise HTTPException(status_code=404, detail="Fixture not found")
            return {"success": True, "data": fixture}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fixtures/{fixture_id}/statistics")
async def get_fixture_statistics(fixture_id: int):
    """Récupère les statistiques d'un match"""
    try:
        async with api_football_client as client:
            stats = await client.get_fixture_statistics(fixture_id)
            return {"success": True, "data": stats}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/head-to-head")
async def get_head_to_head(
    team1_id: int = Query(..., description="ID de l'équipe 1"),
    team2_id: int = Query(..., description="ID de l'équipe 2"),
    last: Optional[int] = Query(None, description="Nombre de matchs")
):
    """Récupère l'historique des confrontations"""
    try:
        async with api_football_client as client:
            h2h = await client.get_head_to_head(team1_id, team2_id, last)
            return {"success": True, "count": len(h2h), "data": h2h}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/standings")
async def get_standings(
    league_id: int = Query(..., description="ID de la ligue"),
    season: int = Query(..., description="Saison"),
    team_id: Optional[int] = Query(None, description="ID de l'équipe")
):
    """Récupère le classement"""
    try:
        async with api_football_client as client:
            standings = await client.get_standings(league_id, season, team_id)
            return {"success": True, "data": standings}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/players/top-scorers")
async def get_top_scorers(
    league_id: int = Query(..., description="ID de la ligue"),
    season: int = Query(..., description="Saison")
):
    """Récupère le classement des buteurs"""
    try:
        async with api_football_client as client:
            scorers = await client.get_top_scorers(league_id, season)
            return {"success": True, "count": len(scorers), "data": scorers}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/players/top-assists")
async def get_top_assists(
    league_id: int = Query(..., description="ID de la ligue"),
    season: int = Query(..., description="Saison")
):
    """Récupère le classement des passeurs"""
    try:
        async with api_football_client as client:
            assists = await client.get_top_assists(league_id, season)
            return {"success": True, "count": len(assists), "data": assists}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/{fixture_id}")
async def get_predictions(fixture_id: int):
    """Récupère les prédictions de l'API pour un match"""
    try:
        async with api_football_client as client:
            predictions = await client.get_predictions(fixture_id)
            if not predictions:
                raise HTTPException(status_code=404, detail="Predictions not found")
            return {"success": True, "data": predictions}
    except APIFootballException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/rate-limiter")
async def get_rate_limiter_stats():
    """Récupère les statistiques du rate limiter"""
    return api_football_client.get_rate_limiter_stats()

@router.get("/stats/cache")
async def get_cache_stats():
    """Récupère les statistiques du cache"""
    return api_football_client.get_cache_stats()

@router.post("/analyze-match")
async def analyze_match_llm(request: LLMRequest):
    """Récupère l'analyse d'un match"""
    return analyze_match(request.team_a, request.team_b)

