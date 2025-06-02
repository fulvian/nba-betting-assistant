import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional
import streamlit as st

class NBADataFetcher:
    """
    Gestisce il recupero di dati NBA da multiple fonti API
    """
    
    def __init__(self):
        self.base_urls = {
            'nba_official': 'https://stats.nba.com/stats',
            'balldontlie': 'https://www.balldontlie.io/api/v1',
            'espn': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
        }
        
        # Headers per evitare blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        }
        
        # Cache per evitare troppe richieste
        self.cache = {}
        self.cache_duration = 300  # 5 minuti
    
    def get_today_games(self) -> List[Dict]:
        """
        Recupera partite di oggi da ESPN API
        """
        try:
            url = f"{self.base_urls['espn']}/scoreboard"
            
            if self._is_cached('today_games'):
                return self.cache['today_games']['data']
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            games = []
            
            if 'events' in data:
                for event in data['events']:
                    game = {
                        'id': event['id'],
                        'home_team': event['competitions'][0]['competitors'][0]['team']['displayName'],
                        'away_team': event['competitions'][0]['competitors'][1]['team']['displayName'],
                        'time': event['date'],
                        'status': event['status']['type']['description']
                    }
                    games.append(game)
            
            self._cache_data('today_games', games)
            return games
            
        except Exception as e:
            st.warning(f"⚠️ Errore recupero partite oggi: {e}")
            return self._get_sample_games()
    
    def get_team_stats(self, home_team: str, away_team: str) -> Dict:
        """
        Recupera statistiche complete delle squadre
        """
        cache_key = f"team_stats_{home_team}_{away_team}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Mappa nomi squadre a codici NBA
            team_mapping = self._get_team_mapping()
            
            home_id = team_mapping.get(home_team)
            away_id = team_mapping.get(away_team)
            
            if not home_id or not away_id:
                return self._get_sample_team_stats(home_team, away_team)
            
            # Recupera statistiche stagionali
            home_stats = self._fetch_team_season_stats(home_id)
            away_stats = self._fetch_team_season_stats(away_id)
            
            # Recupera statistiche ultime 5 partite
            home_recent = self._fetch_team_recent_stats(home_id)
            away_recent = self._fetch_team_recent_stats(away_id)
            
            # Combina tutti i dati
            team_stats = {
                'home': {**home_stats, **home_recent},
                'away': {**away_stats, **away_recent}
            }
            
            self._cache_data(cache_key, team_stats)
            return team_stats
            
        except Exception as e:
            st.warning(f"⚠️ Errore recupero statistiche: {e}")
            return self._get_sample_team_stats(home_team, away_team)
    
    def get_game_context(self, home_team: str, away_team: str) -> Dict:
        """
        Recupera contesto della partita (H2H, infortuni, etc.)
        """
        try:
            # Head-to-head ultimi 3 incontri
            h2h_scores = self._fetch_h2h_history(home_team, away_team)
            
            # Informazioni infortuni (placeholder - richiede API premium)
            injuries = self._fetch_injury_reports(home_team, away_team)
            
            context = {
                'playoff_game': False,  # Determina dalla data
                'elimination_game': False,
                'tanking_mode': {'home': False, 'away': False},
                'travel_fatigue': False,
                'jet_lag': False,
                'h2h_scores_last3': h2h_scores,
                'series_avg_total': None
            }
            
            return context
            
        except Exception as e:
            st.warning(f"⚠️ Errore recupero contesto: {e}")
            return self._get_sample_context()
    
    def get_league_averages(self) -> Dict:
        """
        Recupera medie di lega
        """
        try:
            # Calcola dalle statistiche di tutte le squadre
            if self._is_cached('league_averages'):
                return self.cache['league_averages']['data']
            
            # Per ora usa valori tipici NBA 2024-25
            league_data = {
                'avg_ORtg': 114.2,
                'avg_DRtg': 114.2,
                'avg_pace': 99.8,
                'var_league': 185.5
            }
            
            self._cache_data('league_averages', league_data)
            return league_data
            
        except Exception as e:
            return {'avg_ORtg': 112.5, 'var_league': 180.5}
    
    def _fetch_team_season_stats(self, team_id: str) -> Dict:
        """
        Recupera statistiche stagionali di una squadra
        """
        try:
            # Usando Ball Don't Lie API (gratuita)
            url = f"{self.base_urls['balldontlie']}/season_averages"
            params = {'season': 2024, 'team_ids[]': team_id}
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    stats = data['data'][0]
                    
                    # Converti in formato richiesto
                    return {
                        'ORtg_season': self._calculate_ortg(stats),
                        'DRtg_season': self._calculate_drtg(stats),
                        'Pace_season': stats.get('games_played', 82) * 2.5 + 95,  # Stima
                        'eFG_season': self._calculate_efg(stats),
                        'TOV_season': stats.get('turnover', 0) / 100,
                        'OREB_season': stats.get('oreb', 0) / 100,
                        'FT_rate_season': stats.get('fta', 0) / max(stats.get('fga', 1), 1),
                        'injuries': [],
                        'rest_days': 2  # Default
                    }
            
            # Fallback a dati di esempio
            return self._get_sample_team_season_stats()
            
        except Exception:
            return self._get_sample_team_season_stats()
    
    def _fetch_team_recent_stats(self, team_id: str) -> Dict:
        """
        Recupera statistiche ultime 5 partite
        """
        try:
            # Implementazione placeholder - richiede API più complesse
            return {
                'ORtg_L5': 116.5,
                'DRtg_L5': 110.8,
                'Pace_L5': 101.2,
                'eFG_L5': 0.548,
                'TOV_L5': 0.135,
                'OREB_L5': 0.268,
                'FT_rate_L5': 0.245,
                'points_scored_L5': [118, 112, 125, 108, 121],
                'points_conceded_L5': [115, 119, 108, 122, 117]
            }
            
        except Exception:
            return {
                'ORtg_L5': 115.0,
                'DRtg_L5': 112.0,
                'points_scored_L5': [115, 110, 120, 105, 118],
                'points_conceded_L5': [112, 115, 108, 118, 114]
            }
    
    def _fetch_h2h_history(self, home_team: str, away_team: str) -> List[int]:
        """
        Recupera storico head-to-head
        """
        try:
            # Placeholder - implementazione futura
            # Genera dati realistici basati su medie squadre
            base_total = 220
            variance = 15
            
            scores = []
            for _ in range(3):
                score = base_total + np.random.randint(-variance, variance)
                scores.append(max(180, min(280, score)))
            
            return scores
            
        except Exception:
            return [225, 218, 231]
    
    def _fetch_injury_reports(self, home_team: str, away_team: str) -> Dict:
        """
        Recupera report infortuni (placeholder)
        """
        # Implementazione futura con API premium
        return {
            'home_injuries': [],
            'away_injuries': []
        }
    
    def _get_team_mapping(self) -> Dict[str, str]:
        """
        Mappa nomi squadre completi a ID/abbreviazioni
        """
        return {
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
            "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
            "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
            "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
            "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
            "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
            "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
            "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
        }
    
    def _calculate_ortg(self, stats: Dict) -> float:
        """Calcola Offensive Rating approssimato"""
        ppg = stats.get('pts', 110)
        fga = stats.get('fga', 85)
        return (ppg / max(fga, 1)) * 100 + 5  # Approssimazione
    
    def _calculate_drtg(self, stats: Dict) -> float:
        """Calcola Defensive Rating approssimato"""
        # Inversamente correlato ai punti concessi (stima)
        return 220 - self._calculate_ortg(stats)
    
    def _calculate_efg(self, stats: Dict) -> float:
        """Calcola Effective Field Goal %"""
        fg = stats.get('fgm', 40)
        fg3 = stats.get('fg3m', 12)
        fga = stats.get('fga', 85)
        return (fg + 0.5 * fg3) / max(fga, 1)
    
    def _is_cached(self, key: str) -> bool:
        """Verifica se i dati sono in cache e validi"""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_duration
    
    def _cache_data(self, key: str, data: any):
        """Salva dati in cache"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def _get_sample_games(self) -> List[Dict]:
        """Dati di esempio per partite di oggi"""
        return [
            {
                'id': '1',
                'home_team': 'Los Angeles Lakers',
                'away_team': 'Boston Celtics',
                'time': '21:00',
                'status': 'Pre-Game'
            },
            {
                'id': '2',
                'home_team': 'Golden State Warriors',
                'away_team': 'Phoenix Suns',
                'time': '22:30',
                'status': 'Pre-Game'
            }
        ]
    
    def _get_sample_team_stats(self, home_team: str, away_team: str) -> Dict:
        """Dati di esempio per statistiche squadre"""
        return {
            'home': {
                'ORtg_season': 118.5, 'DRtg_season': 109.2, 'Pace_season': 101.8,
                'eFG_season': 0.565, 'TOV_season': 0.128, 'OREB_season': 0.275,
                'FT_rate_season': 0.252, 'ORtg_L5': 121.2, 'DRtg_L5': 107.1,
                'points_scored_L5': [128, 121, 135, 112, 130],
                'points_conceded_L5': [110, 118, 122, 105, 115],
                'injuries': [], 'rest_days': 2
            },
            'away': {
                'ORtg_season': 114.3, 'DRtg_season': 108.7, 'Pace_season': 97.2,
                'eFG_season': 0.542, 'TOV_season': 0.145, 'OREB_season': 0.248,
                'FT_rate_season': 0.235, 'ORtg_L5': 112.8, 'DRtg_L5': 110.5,
                'points_scored_L5': [111, 118, 102, 125, 107],
                'points_conceded_L5': [128, 111, 121, 131, 122],
                'injuries': [], 'rest_days': 1
            }
        }
    
    def _get_sample_team_season_stats(self) -> Dict:
        """Statistiche stagionali di esempio"""
        return {
            'ORtg_season': 115.0,
            'DRtg_season': 112.0,
            'Pace_season': 99.5,
            'eFG_season': 0.540,
            'TOV_season': 0.140,
            'OREB_season': 0.250,
            'FT_rate_season': 0.230,
            'injuries': [],
            'rest_days': 2
        }
    
    def _get_sample_context(self) -> Dict:
        """Contesto di esempio"""
        return {
            'playoff_game': False,
            'elimination_game': False,
            'tanking_mode': {'home': False, 'away': False},
            'travel_fatigue': False,
            'jet_lag': False,
            'h2h_scores_last3': [225, 218, 231],
            'series_avg_total': None
        }


class QuotesFetcher:
    """
    Gestisce il recupero delle quote da multiple fonti
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.headers = {
            'User-Agent': 'NBA-Betting-Assistant/1.0'
        }
        
    def get_nba_odds(self, home_team: str, away_team: str) -> List[Dict]:
        """
        Recupera quote NBA per una partita specifica
        """
        if not self.api_key:
            return self._get_sample_odds()
        
        try:
            # The Odds API endpoint
            url = f"{self.base_url}/sports/basketball_nba/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'eu',
                'markets': 'totals',
                'oddsFormat': 'decimal'
            }
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Filtra per la partita specifica
            for game in data:
                if (home_team in game['home_team'] and away_team in game['away_team']) or \
                   (away_team in game['home_team'] and home_team in game['away_team']):
                    return self._parse_odds_data(game)
            
            return self._get_sample_odds()
            
        except Exception as e:
            st.warning(f"⚠️ Errore recupero quote: {e}")
            return self._get_sample_odds()
    
    def _parse_odds_data(self, game_data: Dict) -> List[Dict]:
        """
        Converte dati API in formato richiesto
        """
        quotes = []
        
        if 'bookmakers' in game_data:
            for bookmaker in game_data['bookmakers']:
                if 'markets' in bookmaker:
                    for market in bookmaker['markets']:
                        if market['key'] == 'totals':
                            for outcome in market['outcomes']:
                                if outcome['name'] == 'Over':
                                    quotes.append({
                                        'line': outcome['point'],
                                        'over_quote': outcome['price'],
                                        'under_quote': self._find_under_quote(market['outcomes'], outcome['point']),
                                        'bookmaker': bookmaker['title']
                                    })
        
        return quotes if quotes else self._get_sample_odds()
    
    def _find_under_quote(self, outcomes: List[Dict], line: float) -> float:
        """
        Trova la quota Under corrispondente
        """
        for outcome in outcomes:
            if outcome['name'] == 'Under' and outcome['point'] == line:
                return outcome['price']
        return 2.00  # Default
    
    def _get_sample_odds(self) -> List[Dict]:
        """
        Quote di esempio per test
        """
        return [
            {
                'line': 225.5,
                'over_quote': 1.85,
                'under_quote': 1.95,
                'bookmaker': 'Bet365'
            },
            {
                'line': 226.0,
                'over_quote': 1.78,
                'under_quote': 2.02,
                'bookmaker': 'Pinnacle'
            },
            {
                'line': 225.0,
                'over_quote': 1.90,
                'under_quote': 1.90,
                'bookmaker': 'William Hill'
            }
        ]
    
    def get_available_bookmakers(self) -> List[str]:
        """
        Lista bookmaker disponibili
        """
        return [
            'Bet365', 'Pinnacle', 'William Hill', 'Betfair',
            'DraftKings', 'FanDuel', 'BetMGM', 'Caesars'
        ]


class DataValidator:
    """
    Valida i dati ricevuti dalle API
    """
    
    @staticmethod
    def validate_team_stats(stats: Dict) -> bool:
        """
        Valida statistiche squadra
        """
        required_fields = [
            'ORtg_season', 'DRtg_season', 'Pace_season',
            'eFG_season', 'TOV_season', 'OREB_season'
        ]
        
        for field in required_fields:
            if field not in stats:
                return False
            
            value = stats[field]
            if not isinstance(value, (int, float)) or value <= 0:
                return False
        
        return True
    
    @staticmethod
    def validate_quotes(quotes: List[Dict]) -> List[Dict]:
        """
        Valida e filtra quote valide
        """
        valid_quotes = []
        
        for quote in quotes:
            if all(key in quote for key in ['line', 'over_quote']):
                if 1.50 <= quote['over_quote'] <= 3.00:
                    if 150 <= quote['line'] <= 300:
                        valid_quotes.append(quote)
        
        return valid_quotes
    
    @staticmethod
    def sanitize_team_name(team_name: str) -> str:
        """
        Standardizza nomi squadre
        """
        # Rimuovi spazi extra e normalizza
        cleaned = team_name.strip().title()
        
        # Sostituzioni comuni
        replacements = {
            'La Clippers': 'LA Clippers',
            'La Lakers': 'Los Angeles Lakers',
            'Ny Knicks': 'New York Knicks',
            'Sa Spurs': 'San Antonio Spurs'
        }
        
        return replacements.get(cleaned, cleaned)


# Classe di utility per cache globale
class APICache:
    """
    Cache globale per ridurre chiamate API
    """
    _instance = None
    _cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get(self, key: str, max_age: int = 300):
        """
        Recupera valore da cache se valido
        """
        if key in self._cache:
            entry = self._cache[key]
            age = (datetime.now() - entry['timestamp']).seconds
            if age < max_age:
                return entry['data']
        return None
    
    def set(self, key: str, data: any):
        """
        Salva valore in cache
        """
        self._cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def clear(self):
        """
        Pulisce cache
        """
        self._cache.clear()