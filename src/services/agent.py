from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime

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


def analyze_match(team_a: str, team_b: str):
    print(team_a, team_b)
    result = agent.invoke({"team_a": team_a, "team_b": team_b}, context={"team_a": team_a, "team_b": team_b})
    return result