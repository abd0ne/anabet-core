from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime
from typing import Optional
import re
import json

@dataclass
class Context:  
    team_a: str = "team_a"
    team_b: str = "team_b"

@dynamic_prompt
def dynamic_prompt(request: ModelRequest) -> str:  
    team_a = request.runtime.context.team_a
    team_b = request.runtime.context.team_b
    return f"""
    Tu es un expert en analyse de matchs de football.
    
    ÉQUIPE A ({team_a}) - DOMICILE:
    ÉQUIPE B ({team_b}) - EXTÉRIEUR:
    
    ANALYSE DEMANDÉE:
    1. Résultat probable (1-X-2) avec pourcentages
    2. Plus/Moins 2.5 buts avec justification
    3. xG estimé pour chaque équipe
    4. Tirs attendus
    5. Probabilité de clean sheet
    6. Justification statistique complète
    
    Format de réponse en JSON structuré.

    Exemple de réponse:
    {{
        "oneXTwo": {{ // Résultat probable (1_X_2)
            "one": 50, // Probabilité de victoire pour l'équipe A
            "x": 30, // Probabilité de match nul pour l'équipe A
            "two": 20 // Probabilité de victoire pour l'équipe B
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
        "injuries_equipe_a": {{
            "joueur_1": "blessure",
            "joueur_2": "blessure",
            "joueur_3": "blessure"
        }},
        "injuries_equipe_b": {{
            "joueur_1": "blessure",
            "joueur_2": "blessure",
            "joueur_3": "blessure"
        }}
    }}
    """  

# Configuration du modèle
model = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.5,
    base_url="http://localhost:11434"
)

# Créer l'agent
agent = create_agent(
    model=model,
    tools=[],
    middleware=[dynamic_prompt],  
    context_schema=Context
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

def analyze_match(team_a: str, team_b: str):
    print(team_a, team_b)
    result = agent.invoke({"team_a": team_a, "team_b": team_b}, context={"team_a": team_a, "team_b": team_b})
    
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