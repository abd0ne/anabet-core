import os
import time
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
import re
import json
from langchain_tavily import TavilySearch

# Classes pour la structure de réponse
class OneXTwo(BaseModel):
    equipe_a: float = Field(description="Probabilité de victoire pour l'équipe A")
    equipe_x: float = Field(description="Probabilité de match nul")
    equipe_b: float = Field(description="Probabilité de victoire pour l'équipe B")
    justification: str = Field(description="Justification statistique complète")

class TirsAttendus(BaseModel):
    equipe_a: int = Field(description="Nombre de tirs attendus pour l'équipe A")
    equipe_b: int = Field(description="Nombre de tirs attendus pour l'équipe B")

class ProbabiliteCleanSheet(BaseModel):
    equipe_a: float = Field(description="Probabilité de ne pas encaisser de but pour l'équipe A")
    equipe_b: float = Field(description="Probabilité de ne pas encaisser de but pour l'équipe B")
    justification: str = Field(description="Justification statistique complète")

class PourcentagesPlusMoins25Buts(BaseModel):
    plus: float = Field(description="Probabilité de plus de 2.5 buts")
    moins: float = Field(description="Probabilité de moins de 2.5 buts")
    justification: str = Field(description="Justification statistique complète")

class InformationMissing(BaseModel):
    nom: str = Field(description="Nom du joueur ou du joueur blessé ou suspendu")
    position: str = Field(description="Position du joueur ou du joueur blessé ou suspendu")
    etat: str = Field(description="État de la blessure ou de la suspension")
    impact: str = Field(description="Impact sur le match - faible, moyen, fort")    

class InformationsMissing(BaseModel):
    equipe_a: List[InformationMissing] = Field(description="Informations sur les blessures et suspensions pour l'équipe A")
    equipe_b: List[InformationMissing] = Field(description="Informations sur les blessures et suspensions pour l'équipe B")

class AnalyseMatchResponse(BaseModel):
    oneXTwo: OneXTwo = Field(description="Résultat probable (1-X-2) avec pourcentages")
    xg_equipe_a: float = Field(description="xG estimé pour l'équipe A")
    xg_equipe_b: float = Field(description="xG estimé pour l'équipe B")
    tirs_attendus: TirsAttendus = Field(description="Tirs attendus pour chaque équipe")
    probabilite_clean_sheet: ProbabiliteCleanSheet = Field(description="Probabilité de clean sheet")
    pourcentages_plus_moins_2_5_buts: PourcentagesPlusMoins25Buts = Field(
        alias="pourcentages_plus_moins_2.5_buts",
        description="Pourcentages plus/moins 2.5 buts"
    )
    informations_missing: InformationsMissing = Field(description="Informations sur les blessures importantes et les suspensions pour chaque équipe")
    model_config = ConfigDict(populate_by_name=True)

@dataclass
class Context:  
    team_a: str = "team_a"
    team_b: str = "team_b"
    date: str = "date"

@dynamic_prompt
def dynamic_prompt(request: ModelRequest) -> str:  
    team_a = request.runtime.context.team_a
    team_b = request.runtime.context.team_b
    date = request.runtime.context.date

    return f"""
    Tu es un expert en analyse de matchs de football.
    Analyse les informations récentes disponibles sur le match du {date} entre les équipes {team_a} et {team_b}.
    
    IMPORTANT - BASE DE L'ANALYSE:
    Toutes tes analyses DOIVENT se baser UNIQUEMENT sur la saison en 2025/2026 et sur les effectifs actuels des équipes. Prends en compte:
    - Les joueurs disponibles pour ce match spécifique
    - Les blessures et suspensions actuelles
    - Les compositions d'équipe récentes
    - Les statistiques des joueurs présents dans l'effectif actuel
    
    CRITÈRE SPÉCIFIQUE POUR "informations_missing" - À RESPECTER ABSOLUMENT:
    - INCLUSION STRICTE: N'inclus QUE les joueurs qui font ACTUELLEMENT partie de l'effectif de la saison en cours de chaque équipe
    - VÉRIFICATION OBLIGATOIRE: Avant d'inclure un joueur blessé/suspendu, vérifie qu'il fait partie de l'effectif actuel de la saison en cours
    - EXCLUSION ABSOLUE: N'inclus AUCUN joueur qui ne fait plus partie de l'effectif actuel, même s'il était blessé/suspendu dans le passé
    
    RÈGLE GÉNÉRALE: Si tu n'es pas certain qu'un joueur fait partie de l'effectif actuel de la saison en cours, NE L'INCLUS PAS. Mieux vaut une liste incomplète qu'une liste avec des joueurs qui ne font plus partie de l'équipe.
    
    INSTRUCTIONS CRITIQUES - À RESPECTER STRICTEMENT:
    1. RECHERCHE UNIQUE: Utilise l'outil de recherche UNE SEULE FOIS pour TOUTES les informations (équipe A, équipe B, match). Ne fais PAS de recherches multiples.
    2. ARRÊT IMMÉDIAT: Dès que tu as fait CETTE recherche, tu DOIS IMMÉDIATEMENT générer et retourner ta réponse JSON.
    3. INTERDICTION ABSOLUE: Ne fais JAMAIS plus d'une recherche. Ne vérifie pas tes résultats avec une autre recherche.
    4. FORMAT: Ta réponse DOIT être UNIQUEMENT un JSON valide et COMPLET.
    5. RESPECT STRICT DE L'EXEMPLE: Ton JSON DOIT avoir EXACTEMENT les mêmes clés et la même structure que l'exemple ci-dessous. Aucune clé ne doit manquer.
    
    WORKFLOW OBLIGATOIRE (2 étapes seulement):
    Étape 1: Recherche sur internet UNIQUE pour les deux équipes
    Étape 2: Génération du JSON final (conforme à l'exemple) et ARRÊT IMMÉDIAT
    
    ÉQUIPE A ({team_a}) - DOMICILE:
    ÉQUIPE B ({team_b}) - EXTÉRIEUR:
    
    ANALYSE DEMANDÉE:
    1. Résultat probable (1-X-2) avec pourcentages
    2. Plus/Moins 2.5 buts avec justification
    3. xG estimé pour chaque équipe
    4. Tirs attendus
    5. Probabilité de clean sheet
    6. Justification statistique complète
    7. Blessures importantes pour chaque équipe

    Exemple de réponse:
    {{
        "oneXTwo": {{ // Résultat probable (1_X_2)
            "equipe_a": 50, // Probabilité de victoire pour l'équipe A
            "equipe_x": 30, // Probabilité de match nul pour l'équipe A
            "equipe_b": 20, // Probabilité de victoire pour l'équipe B
            "justification": "La probabilité de victoire pour l'équipe A est de 50%, la probabilité de match nul est de 30% et la probabilité de victoire pour l'équipe B est de 20%." // Justification statistique complète
        }},
        "xg_equipe_a": 1.5, // xG estimé pour l'équipe A
        "xg_equipe_b": 1.2, // xG estimé pour l'équipe B
        "tirs_attendus": {{
            "equipe_a": 10, // Nombre de tirs attendus pour l'équipe A
            "equipe_b": 8 // Nombre de tirs attendus pour l'équipe B
        }},
        "probabilite_clean_sheet": {{
            "equipe_a": 50, // Probabilité de ne pas encaisser de but pour l'équipe A   
            "equipe_b": 30, // Probabilité de ne pas encaisser de but pour l'équipe B
            "justification": "La probabilité de clean sheet est de 50% pour l'équipe A et de 30% pour l'équipe B." // Justification statistique complète
        }},
        "pourcentages_plus_moins_2.5_buts": {{
            "plus": 50, // Probabilité de plus de 2.5 buts
            "moins": 30, // Probabilité de moins de 2.5 buts
            "justification": "La probabilité de plus de 2.5 buts est de 50% et la probabilité de moins de 2.5 buts est de 30%." // Justification statistique complète
        }},
        "informations_missing": {{
            "equipe_a": [{{
                "nom": "John Doe",
                "position": "Milieu offensif",
                "etat": "Blessé (tête, 4-6 semaines)",
                "impact": "moyen - explication de l'impact sur le match"
            }},{{
                "nom": "Jean Doe",
                "position": "Défenseur",
                "etat": "Suspension",
                "impact": "fort - explication de l'impact sur le match"
            }}],
            "equipe_b": [{{
                "nom": "Robert Doe",
                "position": "Gardien de but",
                "etat": "Blessé (tête, 1-6 mois)",
                "impact": "moyen - explication de l'impact sur le match"
            }}]
        }},
    }}
    """  

# Configuration du modèle
model = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.3,
    base_url="http://localhost:11434"
)

# Initialize Tavily Search Tool
tavily_search_tool = TavilySearch(
    max_results=1,
    topics=["football", "sports", "news", "injuries", "statistics", "predictions", "odds", "bookmakers", "formations", "lineups", "injuries", "statistics", "predictions", "odds", "bookmakers", "formations", "lineups"],
    api_key=os.getenv("TAVILY_API_KEY"),
    api_url="https://api.tavily.com/v1/search",
)

# Créer l'agent
agent = create_agent(
    model=model,
    tools=[tavily_search_tool],
    middleware=[dynamic_prompt],  
    context_schema=Context,
    response_format=AnalyseMatchResponse
)


def fix_incomplete_json(json_str: str) -> str:
    """Tente de réparer un JSON incomplet en ajoutant les accolades fermantes manquantes"""
    json_str = json_str.strip()
    
    # Compter les accolades ouvertes et fermées
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    # Ajouter les accolades/brackets fermantes manquantes
    missing_braces = open_braces - close_braces
    missing_brackets = open_brackets - close_brackets
    
    # Si le JSON ne se termine pas par } ou ], ajouter les fermetures manquantes
    if not json_str.endswith('}') and not json_str.endswith(']'):
        # Ajouter les brackets fermants d'abord (s'ils sont dans des tableaux)
        if missing_brackets > 0:
            json_str += ']' * missing_brackets
        # Puis les accolades fermantes
        if missing_braces > 0:
            json_str += '}' * missing_braces
    
    return json_str

def extract_json_from_markdown(content: str) -> Optional[dict]:
    """Extrait le JSON d'un bloc de code markdown si présent"""
    # Chercher un bloc de code JSON (```json ... ```)
    json_pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(json_pattern, content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Essayer de réparer le JSON incomplet
            try:
                fixed_json = fix_incomplete_json(json_str)
                return json.loads(fixed_json)
            except (json.JSONDecodeError, Exception):
                pass
    
    # Si pas de bloc markdown, essayer de parser directement le contenu comme JSON
    # ou chercher le premier { et le dernier }
    try:
        # Chercher le premier { et le dernier }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_content = content[start_idx:end_idx+1]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                fixed_json = fix_incomplete_json(json_content)
                return json.loads(fixed_json)
                
        return json.loads(content.strip())
    except (json.JSONDecodeError, Exception):
        pass
    
    # Si aucun JSON valide, retourner None
    return None

def analyze_match(team_a: str, team_b: str, date: str):
    print(team_a, team_b, date)
    # Configuration avec limite de récursion augmentée pour sécurité
    config = {
        "recursion_limit": 30,
        "configurable": {
            "thread_id": f"match_{team_a}_{team_b}_{date}"
        }
    }

    result = agent.invoke(
        {"team_a": team_a, "team_b": team_b, "date": date}, 
        context={"team_a": team_a, "team_b": team_b, "date": date},
        config=config
    )
    
    # Extraire le contenu du message
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        if messages and len(messages) > 0:
            # Récupérer le contenu du dernier message (généralement la réponse de l'IA)
            # messages[-1] est un objet AIMessage, pas un dict
            last_message = messages[-1]
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # Essayer d'extraire le JSON du contenu
            print(f"Contenu brut de la réponse: {content}")
            json_data = extract_json_from_markdown(content)
            if json_data:
                return json_data
            
            print("Impossible d'extraire un JSON valide du contenu.")
            # Sinon, retourner le contenu brut
            return content
    
    # Si la structure est différente, retourner le résultat tel quel
    return result