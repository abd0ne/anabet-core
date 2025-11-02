from datetime import datetime, timedelta
from typing import Optional, Any
import json
import hashlib

class CacheService:
    """Service de cache en mémoire (TODO: en production, utiliser Redis)"""
    
    def __init__(self):
        self._cache: dict[str, dict] = {}
    
    def _generate_key(self, endpoint: str, params: dict) -> str:
        """Génère une clé unique pour le cache"""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{endpoint}:{params_str}".encode()).hexdigest()
    
    def get(self, endpoint: str, params: dict) -> Optional[Any]:
        """Récupère une valeur du cache"""
        key = self._generate_key(endpoint, params)
        
        if key in self._cache:
            cache_entry = self._cache[key]
            if datetime.now() < cache_entry['expires_at']:
                return cache_entry['data']
            else:
                # Cache expiré, le supprimer
                del self._cache[key]
        
        return None
    
    def set(self, endpoint: str, params: dict, data: Any, ttl: int):
        """Stocke une valeur dans le cache"""
        key = self._generate_key(endpoint, params)
        self._cache[key] = {
            'data': data,
            'expires_at': datetime.now() + timedelta(seconds=ttl),
            'created_at': datetime.now()
        }
    
    def clear(self):
        """Vide tout le cache"""
        self._cache.clear()
    
    def clear_expired(self):
        """Supprime les entrées expirées"""
        now = datetime.now()
        expired_keys = [
            key for key, value in self._cache.items()
            if now >= value['expires_at']
        ]
        for key in expired_keys:
            del self._cache[key]
    
    def get_stats(self) -> dict:
        """Retourne des statistiques sur le cache"""
        now = datetime.now()
        valid_entries = sum(
            1 for value in self._cache.values()
            if now < value['expires_at']
        )
        
        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self._cache) - valid_entries
        }

# Instance globale
cache_service = CacheService()