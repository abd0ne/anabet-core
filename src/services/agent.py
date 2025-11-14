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

class ButeurEnVue(BaseModel):
    nom: str = Field(description="Nom du joueur")
    position: str = Field(description="Position du joueur")
    nombre_de_buts: int = Field(description="Nombre de buts marqués par le joueur")

class ButeursEnVue(BaseModel):
    equipe_a: List[ButeurEnVue] = Field(description="Les buteurs les plus en vue pour l'équipe A")
    equipe_b: List[ButeurEnVue] = Field(description="Les buteurs les plus en vue pour l'équipe B")

class InformationsInjuries(BaseModel):
    nom: str = Field(description="Nom du joueur")
    position: str = Field(description="Position du joueur")
    etat: str = Field(description="État de la blessure")
    impact: str = Field(description="Impact sur le match")

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
    informations_injuries: InformationsInjuries = Field(description="Informations sur les blessures importantes pour chaque équipe")
    buteurs_en_vue: ButeursEnVue = Field(description="Les buteurs les plus en vue pour chaque équipe")  
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
    
    INSTRUCTIONS CRITIQUES - À RESPECTER ABSOLUMENT:
    1. LIMITE STRICTE: Utilise l'outil de recherche UNE FOIS maximum. Si tu n'obtiens pas d'informations utiles, procède quand même avec tes connaissances.
    2. ARRÊT IMMÉDIAT: Dès que tu as fait une recherche (ou décidé de ne pas en faire), tu DOIS retourner ta réponse finale en JSON. Ne fais AUCUNE autre action.
    3. FORMAT OBLIGATOIRE: Ta réponse DOIT être uniquement le JSON final, sans texte supplémentaire, sans autre utilisation d'outils.
    4. STRUCTURE: Respecte exactement la structure JSON de l'exemple fourni.
    
    RÉSUMÉ: 1 recherche maximum → puis réponse JSON immédiate → FIN.
    
    ÉQUIPE A ({team_a}) - DOMICILE:
    ÉQUIPE B ({team_b}) - EXTÉRIEUR:
    
    ANALYSE DEMANDÉE:
    1. Résultat probable (1-X-2) avec pourcentages
    2. Plus/Moins 2.5 buts avec justification
    3. xG estimé pour chaque équipe
    4. Tirs attendus
    5. Probabilité de clean sheet
    6. Les buteurs les plus en vue pour chaque équipe
    7. Justification statistique complète
    8. Blessures importantes pour chaque équipe
    
    IMPORTANT: Après avoir collecté les informations nécessaires avec les outils disponibles, tu dois IMMÉDIATEMENT retourner ta réponse finale en JSON. Ne continue pas à utiliser les outils une fois que tu as assez d'informations pour faire l'analyse.
    
    Format de réponse en JSON structuré. Il faut absolument que le JSON soit valide et respectant la structure de l'exemple de réponse.

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
        "buteurs_en_vue": {{
            "equipe_a": [{{
                "nom": "John Doe",
                "position": "Milieu offensif",
                "nombre_de_buts": 10
            }}],
            "equipe_b": [{{
                "nom": "John Doe",
                "position": "Gardien de but",
                "nombre_de_buts": 10
            }}]
        }},
        "informations_injuries": {{
            "equipe_a": [{{
                "nom": "John Doe",
                "position": "Milieu offensif",
                "etat": "Blessé (tête, 4-6 semaines)",
                "impact": "moyen - explication de l'impact sur le match"
            }}],
            "equipe_b": [{{
                "nom": "John Doe",
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
    temperature=0.5,
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


def extract_json_from_markdown(content: str) -> Optional[dict]:
    """Extrait le JSON d'un bloc de code markdown si présent"""
    # Chercher un bloc de code JSON (```json ... ```)
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Si pas de bloc markdown, essayer de parser directement le contenu comme JSON
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass
    
    # Si aucun JSON valide, retourner None
    return None

def analyze_match(team_a: str, team_b: str, date: str):
    print(team_a, team_b, date)
    # Configuration avec limite de récursion augmentée et instructions strictes
    config = {
        "recursion_limit": 50,  # Pas de limite de récursion
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
            json_data = extract_json_from_markdown(content)
            if json_data:
                return json_data
            
            # Sinon, retourner le contenu brut
            return content
    
    # Si la structure est différente, retourner le résultat tel quel
    return result