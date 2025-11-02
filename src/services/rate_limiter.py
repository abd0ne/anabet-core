from datetime import datetime, timedelta
from collections import deque
from typing import Optional

class RateLimiter:
    """Rate limiter pour respecter les limites de l'API"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        """
        Args:
            max_requests: Nombre maximum de requêtes
            time_window: Fenêtre de temps en secondes (défaut: 60s)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque = deque()
    
    def can_make_request(self) -> bool:
        """Vérifie si on peut faire une requête"""
        self._clean_old_requests()
        return len(self.requests) < self.max_requests
    
    def add_request(self):
        """Enregistre une nouvelle requête"""
        self.requests.append(datetime.now())
    
    def _clean_old_requests(self):
        """Supprime les requêtes hors de la fenêtre de temps"""
        cutoff_time = datetime.now() - timedelta(seconds=self.time_window)
        while self.requests and self.requests[0] < cutoff_time:
            self.requests.popleft()
    
    def get_wait_time(self) -> Optional[float]:
        """Retourne le temps d'attente en secondes avant la prochaine requête"""
        if self.can_make_request():
            return None
        
        self._clean_old_requests()
        if not self.requests:
            return None
        
        oldest_request = self.requests[0]
        wait_until = oldest_request + timedelta(seconds=self.time_window)
        wait_seconds = (wait_until - datetime.now()).total_seconds()
        
        return max(0, wait_seconds)
    
    def get_stats(self) -> dict:
        """Retourne des statistiques sur le rate limiting"""
        self._clean_old_requests()
        return {
            'requests_in_window': len(self.requests),
            'max_requests': self.max_requests,
            'remaining_requests': self.max_requests - len(self.requests),
            'wait_time_seconds': self.get_wait_time()
        }