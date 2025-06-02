import streamlit as st
import pandas as pd
import numpy as np
import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path
import joblib
import shutil

class DataManager:
    """
    Gestisce tutti gli aspetti di persistenza e gestione dati
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "nba_assistant.db"
        self.models_dir = self.data_dir / "models"
        self.predictions_dir = self.data_dir / "predictions"
        self.cache_dir = self.data_dir / "cache"
        
        # Crea directories se non esistono
        self._ensure_directories()
        
        # Inizializza database
        self._init_database()
    
    def _ensure_directories(self):
        """Crea tutte le directory necessarie"""
        for directory in [self.data_dir, self.models_dir, self.predictions_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Inizializza database SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tabella predizioni
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    game_id TEXT UNIQUE,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    game_date DATE NOT NULL,
                    predicted_total REAL,
                    actual_total REAL,
                    bet_line REAL,
                    bet_quote REAL,
                    bet_stake REAL,
                    bet_result TEXT,
                    profit REAL,
                    confidence_level TEXT,
                    prediction_data TEXT,
                    status TEXT DEFAULT 'pending'
                )
                """)
                
                # Tabella performance
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_predictions INTEGER DEFAULT 0,
                    total_wins INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    roi REAL DEFAULT 0
                )
                """)
                
                # Tabella configurazioni utente
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Tabella cache API
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    expires_at DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                conn.commit()
                
        except Exception as e:
            st.error(f"Errore inizializzazione database: {e}")
    
    def save_prediction(self, prediction_data: Dict) -> bool:
        """Salva una predizione nel database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                game_info = prediction_data['game_info']
                game_id = f"{game_info['date']}_{game_info['away_team']}_{game_info['home_team']}"
                
                # Estrai dati scommessa se presente
                bet_data = prediction_data.get('recommendations', [{}])[0] if prediction_data.get('recommendations') else {}
                
                cursor.execute("""
                INSERT OR REPLACE INTO predictions (
                    game_id, home_team, away_team, game_date,
                    predicted_total, bet_line, bet_quote, bet_stake,
                    confidence_level, prediction_data, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_id,
                    game_info['home_team'],
                    game_info['away_team'], 
                    game_info['date'],
                    prediction_data.get('media_punti'),
                    bet_data.get('line'),
                    bet_data.get('quota_over'),
                    bet_data.get('stake'),
                    bet_data.get('confidenza', 'NO BET'),
                    json.dumps(prediction_data),
                    'pending'
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            st.error(f"Errore salvataggio predizione: {e}")
            return False
    
    def update_prediction_result(self, game_id: str, actual_total: int, 
                                bet_result: str = None, profit: float = None) -> bool:
        """Aggiorna risultato di una predizione"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calcola profitto se non fornito
                if profit is None and bet_result:
                    cursor.execute("""
                    SELECT bet_stake, bet_quote, bet_result FROM predictions WHERE game_id = ?
                    """, (game_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        stake, quote, _ = result
                        if bet_result == 'win' and stake and quote:
                            profit = stake * (quote - 1)
                        elif bet_result == 'loss' and stake:
                            profit = -stake
                        else:
                            profit = 0
                
                cursor.execute("""
                UPDATE predictions 
                SET actual_total = ?, bet_result = ?, profit = ?, status = 'completed'
                WHERE game_id = ?
                """, (actual_total, bet_result, profit, game_id))
                
                conn.commit()
                
                # Aggiorna statistiche performance
                self._update_performance_stats()
                
                return True
                
        except Exception as e:
            st.error(f"Errore aggiornamento risultato: {e}")
            return False
    
    def get_predictions(self, status: str = None, limit: int = None) -> List[Dict]:
        """Recupera predizioni dal database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM predictions"
                params = []
                
                if status:
                    query += " WHERE status = ?"
                    params.append(status)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                
                columns = [description[0] for description in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    prediction = dict(zip(columns, row))
                    
                    # Deserializza prediction_data se presente
                    if prediction['prediction_data']:
                        try:
                            prediction['prediction_data'] = json.loads(prediction['prediction_data'])
                        except:
                            pass
                    
                    results.append(prediction)
                
                return results
                
        except Exception as e:
            st.error(f"Errore recupero predizioni: {e}")
            return []
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """Calcola statistiche performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Data di inizio
                start_date = (datetime.now() - timedelta(days=days)).date()
                
                # Query statistiche base
                cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as total_wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as total_losses,
                    COUNT(CASE WHEN bet_result = 'push' THEN 1 END) as total_pushes,
                    COALESCE(SUM(profit), 0) as total_profit,
                    COALESCE(SUM(bet_stake), 0) as total_staked
                FROM predictions 
                WHERE game_date >= ? AND status = 'completed'
                """, (start_date,))
                
                result = cursor.fetchone()
                
                if result:
                    total_predictions, total_wins, total_losses, total_pushes, total_profit, total_staked = result
                    
                    # Calcola metriche
                    win_rate = (total_wins / total_predictions * 100) if total_predictions > 0 else 0
                    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
                    
                    # Performance per confidenza
                    cursor.execute("""
                    SELECT 
                        confidence_level,
                        COUNT(*) as count,
                        COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                        COALESCE(SUM(profit), 0) as profit
                    FROM predictions 
                    WHERE game_date >= ? AND status = 'completed' AND confidence_level != 'NO BET'
                    GROUP BY confidence_level
                    """, (start_date,))
                    
                    confidence_stats = {}
                    for row in cursor.fetchall():
                        conf_level, count, wins, profit = row
                        confidence_stats[conf_level] = {
                            'count': count,
                            'wins': wins,
                            'win_rate': (wins / count * 100) if count > 0 else 0,
                            'profit': profit
                        }
                    
                    return {
                        'total_predictions': total_predictions,
                        'total_wins': total_wins,
                        'total_losses': total_losses,
                        'total_pushes': total_pushes,
                        'total_profit': total_profit,
                        'total_staked': total_staked,
                        'win_rate': win_rate,
                        'roi': roi,
                        'confidence_breakdown': confidence_stats
                    }
                
                return {}
                
        except Exception as e:
            st.error(f"Errore calcolo performance: {e}")
            return {}
    
    def get_pending_predictions(self) -> List[Dict]:
        """Recupera predizioni in attesa di risultato"""
        return self.get_predictions(status='pending')
    
    def export_data(self, format: str = 'csv') -> Union[str, bytes]:
        """Esporta dati in vari formati"""
        try:
            predictions = self.get_predictions()
            
            if not predictions:
                return ""
            
            # Crea DataFrame
            df_data = []
            for pred in predictions:
                row = {
                    'Data': pred['game_date'],
                    'Partita': f"{pred['away_team']} @ {pred['home_team']}",
                    'Punti_Predetti': pred['predicted_total'],
                    'Punti_Reali': pred['actual_total'],
                    'Linea_Scommessa': pred['bet_line'],
                    'Quota': pred['bet_quote'],
                    'Stake': pred['bet_stake'],
                    'Risultato': pred['bet_result'],
                    'Profitto': pred['profit'],
                    'Confidenza': pred['confidence_level'],
                    'Status': pred['status']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            if format == 'csv':
                return df.to_csv(index=False)
            elif format == 'excel':
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_sheet(writer, sheet_name='Predizioni', index=False)
                return buffer.getvalue()
            elif format == 'json':
                return df.to_json(orient='records', indent=2)
            
        except Exception as e:
            st.error(f"Errore esportazione: {e}")
            return ""
    
    def save_user_config(self, key: str, value: str) -> bool:
        """Salva configurazione utente"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO user_config (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, value))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Errore salvataggio config: {e}")
            return False
    
    def get_user_config(self, key: str, default: str = None) -> str:
        """Recupera configurazione utente"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM user_config WHERE key = ?", (key,))
                result = cursor.fetchone()
                return result[0] if result else default
        except:
            return default
    
    def cache_api_data(self, cache_key: str, data: Dict, expires_hours: int = 24) -> bool:
        """Cache dati API con scadenza"""
        try:
            expires_at = datetime.now() + timedelta(hours=expires_hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO api_cache (cache_key, data, expires_at)
                VALUES (?, ?, ?)
                """, (cache_key, json.dumps(data), expires_at))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Errore cache API: {e}")
            return False
    
    def get_cached_api_data(self, cache_key: str) -> Optional[Dict]:
        """Recupera dati API da cache se validi"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT data FROM api_cache 
                WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                """, (cache_key,))
                
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
                
        except Exception:
            pass
        
        return None
    
    def cleanup_expired_cache(self) -> None:
        """Pulisce cache scaduta"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM api_cache WHERE expires_at <= CURRENT_TIMESTAMP")
                conn.commit()
        except Exception:
            pass
    
    def backup_database(self, backup_path: str = None) -> bool:
        """Crea backup del database"""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.data_dir / f"backup_{timestamp}.db"
            
            shutil.copy2(self.db_path, backup_path)
            return True
            
        except Exception as e:
            st.error(f"Errore backup: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """Ripristina database da backup"""
        try:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, self.db_path)
                return True
            return False
        except Exception as e:
            st.error(f"Errore ripristino: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Statistiche database"""
        try:
            stats = {
                'database_size': os.path.getsize(self.db_path) / 1024 / 1024,  # MB
                'total_predictions': 0,
                'pending_predictions': 0,
                'completed_predictions': 0
            }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM predictions")
                stats['total_predictions'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE status = 'pending'")
                stats['pending_predictions'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE status = 'completed'")
                stats['completed_predictions'] = cursor.fetchone()[0]
            
            return stats
            
        except Exception:
            return {}
    
    def _update_performance_stats(self) -> None:
        """Aggiorna statistiche performance giornaliere"""
        try:
            today = datetime.now().date()
            stats = self.get_performance_stats(days=1)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO performance (
                    date, total_predictions, total_wins, total_profit, win_rate, roi
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    today,
                    stats.get('total_predictions', 0),
                    stats.get('total_wins', 0), 
                    stats.get('total_profit', 0),
                    stats.get('win_rate', 0),
                    stats.get('roi', 0)
                ))
                conn.commit()
                
        except Exception:
            pass


class ModelManager:
    """
    Gestisce modelli ML e loro versioning
    """
    
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.current_version = self._get_current_version()
    
    def save_model(self, model, model_name: str, version: str = None) -> bool:
        """Salva modello con versioning"""
        try:
            if not version:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            model_path = self.models_dir / f"{model_name}_v{version}.joblib"
            joblib.dump(model, model_path)
            
            # Aggiorna current version
            self._update_current_version(model_name, version)
            
            return True
            
        except Exception as e:
            st.error(f"Errore salvataggio modello: {e}")
            return False
    
    def load_model(self, model_name: str, version: str = None):
        """Carica modello specifico"""
        try:
            if not version:
                version = self.current_version.get(model_name)
            
            if not version:
                return None
            
            model_path = self.models_dir / f"{model_name}_v{version}.joblib"
            
            if model_path.exists():
                return joblib.load(model_path)
            
            return None
            
        except Exception as e:
            st.error(f"Errore caricamento modello: {e}")
            return None
    
    def list_models(self) -> Dict:
        """Lista modelli disponibili"""
        models = {}
        
        for model_file in self.models_dir.glob("*.joblib"):
            parts = model_file.stem.split('_v')
            if len(parts) == 2:
                model_name, version = parts
                if model_name not in models:
                    models[model_name] = []
                models[model_name].append(version)
        
        return models
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """Elimina versione specifica del modello"""
        try:
            model_path = self.models_dir / f"{model_name}_v{version}.joblib"
            
            if model_path.exists():
                model_path.unlink()
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Errore eliminazione modello: {e}")
            return False
    
    def _get_current_version(self) -> Dict:
        """Recupera versioni correnti dei modelli"""
        version_file = self.models_dir / "current_versions.json"
        
        if version_file.exists():
            try:
                with open(version_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {}
    
    def _update_current_version(self, model_name: str, version: str) -> None:
        """Aggiorna versione corrente di un modello"""
        version_file = self.models_dir / "current_versions.json"
        
        current_versions = self._get_current_version()
        current_versions[model_name] = version
        
        try:
            with open(version_file, 'w') as f:
                json.dump(current_versions, f, indent=2)
        except Exception:
            pass


class FileManager:
    """
    Gestisce upload e download di file
    """
    
    @staticmethod
    def save_uploaded_file(uploaded_file, save_dir: str = "data/uploads") -> str:
        """Salva file caricato"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = save_path / filename
        
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return str(file_path)
        except Exception as e:
            st.error(f"Errore salvataggio file: {e}")
            return None
    
    @staticmethod
    def create_download_link(data: Union[str, bytes], filename: str, 
                           mime_type: str = "text/plain") -> None:
        """Crea link di download"""
        st.download_button(
            label=f"📥 Download {filename}",
            data=data,
            file_name=filename,
            mime=mime_type,
            use_container_width=True
        )
    
    @staticmethod
    def validate_json_file(file_content: str) -> bool:
        """Valida file JSON"""
        try:
            json.loads(file_content)
            return True
        except:
            return False