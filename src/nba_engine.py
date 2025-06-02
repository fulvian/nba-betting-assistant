"""
NBA ENGINE per Streamlit - Versione ottimizzata del sistema completo
Adattato dalla versione originale per integrazione con la dashboard web
"""

import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings('ignore')

class NBAStreamlitEngine:
    """
    Versione Streamlit del sistema NBA completo
    Ottimizzata per performance e user experience
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_importance = {}
        
        # Features essenziali per il modello
        self.core_features = [
            'HOME_ORtg_sAvg', 'HOME_DRtg_sAvg', 'HOME_PACE',
            'AWAY_ORtg_sAvg', 'AWAY_DRtg_sAvg', 'AWAY_PACE',
            'HOME_eFG_PCT_sAvg', 'HOME_TOV_PCT_sAvg', 'HOME_OREB_PCT_sAvg', 'HOME_FT_RATE_sAvg',
            'AWAY_eFG_PCT_sAvg', 'AWAY_TOV_PCT_sAvg', 'AWAY_OREB_PCT_sAvg', 'AWAY_FT_RATE_sAvg',
            'HOME_ORtg_L5Avg', 'HOME_DRtg_L5Avg', 
            'AWAY_ORtg_L5Avg', 'AWAY_DRtg_L5Avg',
            'GAME_PACE'
        ]
    
    @st.cache_data
    def load_or_train_models(_self, force_retrain: bool = False) -> bool:
        """
        Carica modelli esistenti o addestra nuovi
        """
        models_path = "data/models"
        
        if not force_retrain and os.path.exists(models_path):
            try:
                return _self._load_existing_models(models_path)
            except Exception as e:
                st.warning(f"Errore caricamento modelli: {e}. Riaddestramento...")
        
        # Addestra nuovi modelli
        return _self._train_new_models()
    
    def _load_existing_models(self, models_path: str) -> bool:
        """Carica modelli pre-addestrati"""
        try:
            # Carica metadata
            metadata_path = os.path.join(models_path, "metadata.joblib")
            if not os.path.exists(metadata_path):
                return False
            
            metadata = joblib.load(metadata_path)
            self.features = metadata['features']
            
            # Carica modelli
            for name in metadata['model_names']:
                model_path = os.path.join(models_path, f"{name}_model.joblib")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
            
            # Carica scalers
            for name in metadata['scaler_names']:
                scaler_path = os.path.join(models_path, f"{name}_scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scalers[name] = joblib.load(scaler_path)
            
            self.feature_importance = metadata.get('feature_importance', {})
            self.is_trained = True
            
            st.success("✅ Modelli ML caricati dalla cache")
            return True
            
        except Exception as e:
            st.error(f"Errore caricamento modelli: {e}")
            return False
    
    def _train_new_models(self) -> bool:
        """Addestra nuovi modelli con progress bar"""
        try:
            # Controlla se esiste dataset
            dataset_path = "data/nba_game_data_with_rolling_features_final.csv"
            
            if not os.path.exists(dataset_path):
                # Genera dataset di esempio per demo
                st.warning("⚠️ Dataset principale non trovato. Genero dati di esempio...")
                self._generate_sample_dataset(dataset_path)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Carica dati
            status_text.text("📊 Caricamento dataset...")
            progress_bar.progress(10)
            
            df = pd.read_csv(dataset_path)
            
            # Preprocessing
            status_text.text("🔧 Preprocessing dati...")
            progress_bar.progress(20)
            
            df = self._preprocess_data(df)
            
            if len(df) < 100:
                st.error("❌ Dataset troppo piccolo per addestramento")
                return False
            
            # Prepara features
            status_text.text("🎯 Preparazione features...")
            progress_bar.progress(30)
            
            X, y = self._prepare_features(df)
            
            # Split dati
            status_text.text("✂️ Split train/test...")
            progress_bar.progress(40)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Addestramento modelli
            models_config = {
                'rf': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
                'gb': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
                'ridge': Ridge(alpha=1.0, random_state=42)
            }
            
            for i, (name, model) in enumerate(models_config.items()):
                status_text.text(f"🤖 Addestramento {name.upper()}...")
                progress_bar.progress(50 + i * 15)
                
                if name == 'ridge':
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    self.scalers[name] = scaler
                else:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                
                model.fit(X_train_scaled, y_train)
                
                # Valutazione
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.models[name] = model
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(X.columns, model.feature_importances_))
                    self.feature_importance[name] = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                st.write(f"📊 {name.upper()}: MAE={mae:.2f}, R²={r2:.3f}")
            
            # Salva modelli
            status_text.text("💾 Salvataggio modelli...")
            progress_bar.progress(95)
            
            self._save_models(X.columns.tolist())
            
            status_text.text("✅ Addestramento completato!")
            progress_bar.progress(100)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            st.error(f"❌ Errore durante addestramento: {e}")
            return False
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing base del dataset"""
        # Filtra dati validi
        df = df.dropna(subset=['TOTAL_SCORE'])
        df = df[(df['TOTAL_SCORE'] >= 150) & (df['TOTAL_SCORE'] <= 300)]
        
        # Features ingegnerizzate
        if all(col in df.columns for col in ['HOME_PACE', 'AWAY_PACE']):
            df['PACE_DIFFERENTIAL'] = df['HOME_PACE'] - df['AWAY_PACE']
            df['ESTIMATED_PACE'] = (df['HOME_PACE'] + df['AWAY_PACE']) / 2
            df['GAME_PACE'] = df['ESTIMATED_PACE']
        
        if all(col in df.columns for col in ['HOME_ORtg_sAvg', 'AWAY_DRtg_sAvg']):
            df['HOME_MATCHUP_ADV'] = df['HOME_ORtg_sAvg'] - df['AWAY_DRtg_sAvg']
            df['AWAY_MATCHUP_ADV'] = df['AWAY_ORtg_sAvg'] - df['HOME_DRtg_sAvg']
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara features per ML"""
        # Features disponibili
        available_features = [f for f in self.core_features if f in df.columns]
        
        # Aggiungi features ingegnerizzate se disponibili
        engineered_features = ['PACE_DIFFERENTIAL', 'ESTIMATED_PACE', 'HOME_MATCHUP_ADV', 'AWAY_MATCHUP_ADV']
        available_features.extend([f for f in engineered_features if f in df.columns])
        
        # Pulisci dati
        df_clean = df.dropna(subset=available_features, thresh=int(len(available_features) * 0.8))
        df_clean[available_features] = df_clean[available_features].fillna(df_clean[available_features].median())
        
        X = df_clean[available_features]
        y = df_clean['TOTAL_SCORE']
        
        return X, y
    
    def _save_models(self, features: List[str]) -> None:
        """Salva modelli e metadata"""
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Salva modelli
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(models_dir, f"{name}_model.joblib"))
        
        # Salva scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(models_dir, f"{name}_scaler.joblib"))
        
        # Metadata
        metadata = {
            'features': features,
            'model_names': list(self.models.keys()),
            'scaler_names': list(self.scalers.keys()),
            'feature_importance': self.feature_importance,
            'created_at': datetime.now().isoformat()
        }
        joblib.dump(metadata, os.path.join(models_dir, "metadata.joblib"))
    
    def _generate_sample_dataset(self, output_path: str) -> None:
        """Genera dataset di esempio per demo"""
        np.random.seed(42)
        
        # Genera 1000 partite di esempio
        n_games = 1000
        
        data = []
        for i in range(n_games):
            # Statistiche realistiche NBA
            home_ortg = np.random.normal(115, 8)
            away_ortg = np.random.normal(115, 8)
            home_drtg = np.random.normal(112, 6)
            away_drtg = np.random.normal(112, 6)
            home_pace = np.random.normal(100, 4)
            away_pace = np.random.normal(100, 4)
            
            # Calcola punteggio approssimativo
            game_pace = (home_pace + away_pace) / 2
            home_points = (home_ortg / 100) * game_pace
            away_points = (away_ortg / 100) * game_pace
            total_score = home_points + away_points
            
            # Aggiungi rumore
            total_score += np.random.normal(0, 12)
            total_score = max(160, min(280, total_score))
            
            row = {
                'GAME_ID': f"SAMPLE_{i:04d}",
                'HOME_ORtg_sAvg': home_ortg,
                'HOME_DRtg_sAvg': home_drtg,
                'HOME_PACE': home_pace,
                'AWAY_ORtg_sAvg': away_ortg,
                'AWAY_DRtg_sAvg': away_drtg,
                'AWAY_PACE': away_pace,
                'HOME_eFG_PCT_sAvg': np.random.normal(0.54, 0.03),
                'HOME_TOV_PCT_sAvg': np.random.normal(0.14, 0.02),
                'HOME_OREB_PCT_sAvg': np.random.normal(0.26, 0.04),
                'HOME_FT_RATE_sAvg': np.random.normal(0.24, 0.05),
                'AWAY_eFG_PCT_sAvg': np.random.normal(0.54, 0.03),
                'AWAY_TOV_PCT_sAvg': np.random.normal(0.14, 0.02),
                'AWAY_OREB_PCT_sAvg': np.random.normal(0.26, 0.04),
                'AWAY_FT_RATE_sAvg': np.random.normal(0.24, 0.05),
                'HOME_ORtg_L5Avg': home_ortg + np.random.normal(0, 3),
                'HOME_DRtg_L5Avg': home_drtg + np.random.normal(0, 3),
                'AWAY_ORtg_L5Avg': away_ortg + np.random.normal(0, 3),
                'AWAY_DRtg_L5Avg': away_drtg + np.random.normal(0, 3),
                'GAME_PACE': game_pace,
                'TOTAL_SCORE': total_score
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Crea directory se non esiste
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        st.info(f"📊 Dataset di esempio generato: {len(df)} partite")


# Funzioni principali del sistema (Livelli 1-2-3)

def execute_livello1_streamlit(game_data: Dict) -> Dict:
    """
    LIVELLO 1 - Versione ottimizzata per Streamlit
    """
    with st.expander("🎯 Dettagli Livello 1 - Stima Punteggio Base", expanded=False):
        st.write("**Fase 1.1: Calcolo Baseline**")
        
        team_stats = game_data['team_stats']
        context = game_data.get('context', {})
        league = game_data.get('league_data', {})
        quotes = game_data['quotes']
        
        # Calcoli baseline
        pace_previsto_base = (team_stats['home']['Pace_season'] + team_stats['away']['Pace_season']) / 2
        lg_avg_ortg = league.get('avg_ORtg', 112.5)
        
        ortg_previsto_A = (team_stats['home']['ORtg_season'] * team_stats['away']['DRtg_season']) / lg_avg_ortg
        ortg_previsto_B = (team_stats['away']['ORtg_season'] * team_stats['home']['DRtg_season']) / lg_avg_ortg
        
        punti_previsti_A_base = (ortg_previsto_A / 100) * pace_previsto_base
        punti_previsti_B_base = (ortg_previsto_B / 100) * pace_previsto_base
        punteggio_totale_baseline = punti_previsti_A_base + punti_previsti_B_base
        
        # Mostra metriche baseline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pace Previsto", f"{pace_previsto_base:.1f}")
        with col2:
            st.metric("ORtg Casa", f"{ortg_previsto_A:.1f}")
        with col3:
            st.metric("ORtg Ospite", f"{ortg_previsto_B:.1f}")
        
        st.write(f"**🎯 Punteggio Baseline: {punteggio_totale_baseline:.1f}**")
        
        st.write("**Fase 1.2: Aggiustamenti Situazionali**")
        
        # Aggiustamenti
        aggiustamento_netto = 0.0
        pace_aggiustato = pace_previsto_base
        adjustments = []
        
        # Forma recente
        if all(key in team_stats['home'] for key in ['ORtg_L5', 'DRtg_L5']):
            delta_ortg_A = (team_stats['home']['ORtg_L5'] - team_stats['home']['ORtg_season']) - \
                           (team_stats['away']['DRtg_L5'] - team_stats['away']['DRtg_season'])
            delta_ortg_B = (team_stats['away']['ORtg_L5'] - team_stats['away']['ORtg_season']) - \
                           (team_stats['home']['DRtg_L5'] - team_stats['home']['DRtg_season'])
            
            aggiustamento_forma = ((delta_ortg_A + delta_ortg_B) / 2) * (pace_aggiustato / 100) * 0.5
            aggiustamento_netto += aggiustamento_forma
            
            if aggiustamento_forma != 0:
                adjustments.append(f"Forma recente: {aggiustamento_forma:+.1f}")
        
        # Scontri diretti
        if 'h2h_scores_last3' in context and context['h2h_scores_last3']:
            h2h_scores = context['h2h_scores_last3']
            media_punti_h2h = np.mean(h2h_scores)
            aggiustamento_h2h = (media_punti_h2h - punteggio_totale_baseline) * 0.15
            aggiustamento_netto += aggiustamento_h2h
            
            if aggiustamento_h2h != 0:
                adjustments.append(f"H2H (media {media_punti_h2h:.0f}): {aggiustamento_h2h:+.1f}")
        
        # Infortuni
        home_injuries = team_stats['home'].get('injuries', [])
        away_injuries = team_stats['away'].get('injuries', [])
        if home_injuries or away_injuries:
            aggiustamento_infortuni = -(len(home_injuries) + len(away_injuries)) * 2
            aggiustamento_netto += aggiustamento_infortuni
            
            if aggiustamento_infortuni != 0:
                adjustments.append(f"Infortuni ({len(home_injuries)+len(away_injuries)}): {aggiustamento_infortuni:+.1f}")
        
        # Back-to-back
        home_rest = team_stats['home'].get('rest_days', 2)
        away_rest = team_stats['away'].get('rest_days', 2)
        b2b_adjustment = 0
        
        if home_rest <= 1 and away_rest > 1:
            b2b_adjustment = -1.5
        elif away_rest <= 1 and home_rest > 1:
            b2b_adjustment = -1.5
        elif home_rest <= 1 and away_rest <= 1:
            b2b_adjustment = -1.0
        
        if b2b_adjustment != 0:
            aggiustamento_netto += b2b_adjustment
            adjustments.append(f"Back-to-back: {b2b_adjustment:+.1f}")
        
        # Mostra aggiustamenti
        if adjustments:
            for adj in adjustments:
                st.write(f"• {adj}")
        else:
            st.write("• Nessun aggiustamento applicato")
        
        punteggio_totale_pre_failsafe = punteggio_totale_baseline + aggiustamento_netto
        
        st.write("**Fase 1.3: Fail-Safe Mercato**")
        
        # Fail-safe con quote
        quote_valide = [q for q in quotes if 1.70 <= q['over_quote'] <= 1.95]
        if quote_valide:
            linea_centrale_info = min(quote_valide, 
                                    key=lambda x: abs(x['over_quote'] - x.get('under_quote', 2.0)))
            linea_centrale = linea_centrale_info['line']
            
            dev_rel = abs(punteggio_totale_pre_failsafe - linea_centrale) / linea_centrale
            
            if dev_rel > 0.05:
                media_punti_l1_finale = (punteggio_totale_pre_failsafe + linea_centrale) / 2
                bias_dinamico = media_punti_l1_finale - punteggio_totale_pre_failsafe
                st.write(f"🔧 Fail-safe applicato: media con linea {linea_centrale}")
            else:
                media_punti_l1_finale = punteggio_totale_pre_failsafe
                bias_dinamico = 0.0
                st.write("✅ Fail-safe non necessario")
        else:
            linea_centrale = 225.0
            media_punti_l1_finale = punteggio_totale_pre_failsafe
            bias_dinamico = 0.0
        
        # Risultato finale
        st.success(f"🎯 **Media Punti L1 Finale: {media_punti_l1_finale:.1f}**")
    
    return {
        'Media_punti_stimati_L1_finale': media_punti_l1_finale,
        'Punteggio_totale_baseline': punteggio_totale_baseline,
        'Aggiustamento_Netto_Situazionale': aggiustamento_netto,
        'linea_centrale_bookmaker': linea_centrale,
        'Bias_dinamico_L1': bias_dinamico,
        'Pace_finale_stimato_L1': pace_aggiustato
    }


def calculate_sd_final_streamlit(game_data: Dict, livello1_result: Dict) -> Dict:
    """
    LIVELLO 2 - Calcolo Deviazione Standard per Streamlit
    """
    with st.expander("📊 Dettagli Livello 2 - Deviazione Standard", expanded=False):
        team_stats = game_data['team_stats']
        context = game_data.get('context', {})
        league = game_data.get('league_data', {})
        
        st.write("**Step 3.1: Pooled Variance del Matchup**")
        
        # Calcola varianze
        home_scored_L5 = team_stats['home'].get('points_scored_L5', [])
        away_conceded_L5 = team_stats['away'].get('points_conceded_L5', [])
        h2h_totals = context.get('h2h_scores_last3', [])
        
        var_A5 = np.var(home_scored_L5, ddof=1) if len(home_scored_L5) > 1 else 25.0
        var_B5 = np.var(away_conceded_L5, ddof=1) if len(away_conceded_L5) > 1 else 25.0
        var_H3 = np.var(h2h_totals, ddof=1) if len(h2h_totals) > 1 else 30.0
        
        var_matchup = (4 * var_A5 + 4 * var_B5 + 2 * var_H3) / 10
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("var_A5", f"{var_A5:.1f}")
        with col2:
            st.metric("var_B5", f"{var_B5:.1f}")
        with col3:
            st.metric("var_H3", f"{var_H3:.1f}")
        with col4:
            st.metric("var_matchup", f"{var_matchup:.1f}")
        
        st.write("**Step 3.2: Fusione con Varianza Lega**")
        
        var_league = league.get('var_league', 180.5)
        alpha = 0.75
        var_weighted = (alpha * var_matchup) + ((1 - alpha) * var_league)
        
        st.write(f"• var_league: {var_league:.1f}")
        st.write(f"• alpha: {alpha}")
        st.write(f"• **var_weighted: {var_weighted:.1f}**")
        
        st.write("**Step 3.3: Varianza Contestuale**")
        
        sigma2_context_raw = 0
        conditions_active = []
        
        # Condizioni che aumentano varianza
        if context.get('playoff_game', False):
            sigma2_context_raw += 15
            conditions_active.append("Playoff (+15)")
        
        home_rest = team_stats['home'].get('rest_days', 2)
        away_rest = team_stats['away'].get('rest_days', 2)
        if home_rest <= 1 or away_rest <= 1:
            sigma2_context_raw += 10
            conditions_active.append("Back-to-back (+10)")
        
        tanking = context.get('tanking_mode', {})
        if tanking.get('home', False) or tanking.get('away', False):
            sigma2_context_raw += 20
            conditions_active.append("Tanking (+20)")
        
        # Infortuni significativi
        home_injuries = team_stats['home'].get('injuries', [])
        away_injuries = team_stats['away'].get('injuries', [])
        if len(home_injuries) + len(away_injuries) >= 5:
            sigma2_context_raw += 10
            conditions_active.append("Molti infortuni (+10)")
        
        sigma2_context = max(5, min(sigma2_context_raw, 35))
        
        if conditions_active:
            for condition in conditions_active:
                st.write(f"• {condition}")
        else:
            st.write("• Nessuna condizione speciale")
        
        st.write(f"**σ²_context: {sigma2_context:.1f}**")
        
        # Calcolo finale
        var_finale = var_weighted + sigma2_context
        factor_adj = 1.07 if context.get('playoff_game', False) else 1.00
        sd_final = np.sqrt(var_finale) * factor_adj
        
        st.success(f"🎯 **Deviazione Standard Finale: {sd_final:.1f}**")
    
    return {
        'var_matchup': var_matchup,
        'var_weighted': var_weighted,
        'sigma2_context': sigma2_context,
        'var_finale': var_finale,
        'sd_final': sd_final
    }


def execute_monte_carlo_streamlit(media_punti_l1: float, sd_final: float, 
                                 quote_valide: List[Dict], n_simulations: int = 10000) -> Dict:
    """
    LIVELLO 2 - Simulazione Monte Carlo per Streamlit
    """
    with st.expander("🎲 Dettagli Monte Carlo - Simulazione", expanded=False):
        st.write(f"**Parametri:** μ={media_punti_l1:.1f}, σ={sd_final:.1f}, sim={n_simulations:,}")
        
        # Simulazioni
        simulated_scores = np.random.normal(media_punti_l1, sd_final, n_simulations)
        
        # Calcola probabilità per ogni linea
        monte_carlo_results = {}
        
        for quote in quote_valide:
            line = quote['line']
            p_over = np.sum(simulated_scores > line) / n_simulations
            p_under = 1 - p_over
            
            monte_carlo_results[line] = {
                'P_MC_Over': p_over,
                'P_MC_Under': p_under
            }
            
            st.write(f"• Linea {line}: P(Over) = {p_over:.3f} ({p_over*100:.1f}%)")
        
        # Grafico distribuzione
        import plotly.graph_objects as go
        
        x = np.linspace(media_punti_l1 - 3*sd_final, media_punti_l1 + 3*sd_final, 1000)
        y = stats.norm.pdf(x, media_punti_l1, sd_final)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, fill='tonexty', name='Distribuzione'))
        
        for quote in quote_valide:
            fig.add_vline(x=quote['line'], line_dash="dash", 
                         annotation_text=f"Linea {quote['line']}")
        
        fig.update_layout(
            title="Distribuzione Probabilità Punteggio",
            xaxis_title="Punti Totali",
            yaxis_title="Densità",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    return monte_carlo_results


def execute_livello3_ml_streamlit(media_punti_l1: float, team_stats: Dict) -> Optional[Dict]:
    """
    LIVELLO 3 - Machine Learning per Streamlit
    """
    with st.expander("🤖 Dettagli Livello 3 - Machine Learning", expanded=False):
        try:
            # Inizializza engine
            engine = NBAStreamlitEngine()
            
            # Carica/addestra modelli
            if not engine.load_or_train_models():
                st.error("❌ Impossibile caricare modelli ML")
                return None
            
            # Prepara features
            features = prepare_features_from_team_stats_streamlit(team_stats)
            X_input = dict_to_feature_vector_streamlit(features, engine.features)
            
            # Predizioni
            predictions = []
            model_results = {}
            
            for model_name, model in engine.models.items():
                if model_name in engine.scalers:
                    X_scaled = engine.scalers[model_name].transform([X_input])
                else:
                    X_scaled = [X_input]
                
                pred = model.predict(X_scaled)[0]
                predictions.append(pred)
                model_results[model_name] = pred
                
                st.write(f"• {model_name.upper()}: {pred:.1f} punti")
            
            # Media e fusione
            avg_ml_prediction = np.mean(predictions)
            mu_final_fused = (media_punti_l1 + avg_ml_prediction) / 2
            
            st.write(f"**Media ML:** {avg_ml_prediction:.1f}")
            st.write(f"**μ finale (L1+ML):** {mu_final_fused:.1f}")
            
            return {
                'model_predictions': model_results,
                'avg_ml_prediction': avg_ml_prediction,
                'mu_final_fused': mu_final_fused,
                'qualita_dato_ml': 'Alta' if len(predictions) >= 3 else 'Media',
                'livello3_attivo': True
            }
            
        except Exception as e:
            st.error(f"❌ Errore ML: {e}")
            return None


def execute_value_bet_analysis_streamlit(monte_carlo_results: Dict, livello3_results: Optional[Dict],
                                       quotes: List[Dict], bankroll: float = 100.0) -> List[Dict]:
    """
    STEP 4-6: Analisi Value Bet per Streamlit
    """
    with st.expander("💡 Dettagli Value Bet Analysis", expanded=False):
        scommesse_valide = []
        
        for quote in quotes:
            line = quote['line']
            quota_over = quote['over_quote']
            
            if not (1.70 <= quota_over <= 1.95):
                continue
            
            st.write(f"**Analisi linea {line} @ {quota_over}**")
            
            # Probabilità
            p_mc_over = monte_carlo_results[line]['P_MC_Over']
            
            if livello3_results and livello3_results['livello3_attivo']:
                mu_fused = livello3_results['mu_final_fused']
                p_ml_over = 1 - stats.norm.cdf(line, mu_fused, 12.0)
                probabilita_finale = (p_mc_over + p_ml_over) / 2
                st.write(f"• P(Over) MC: {p_mc_over:.3f}, ML: {p_ml_over:.3f}")
            else:
                probabilita_finale = p_mc_over
                st.write(f"• P(Over) MC: {p_mc_over:.3f}")
            
            # Value bet
            prob_implicita = 1 / quota_over
            edge_value = (quota_over * probabilita_finale) - 1
            
            st.write(f"• Edge: {edge_value:.3f} ({edge_value*100:.1f}%)")
            st.write(f"• Prob finale: {probabilita_finale:.3f} ({probabilita_finale*100:.1f}%)")
            
            # Criteri
            criterio_edge = edge_value > 0.10
            criterio_prob = probabilita_finale > 0.60
            criterio_quota = 1.70 <= quota_over <= 1.80
            
            if criterio_edge and criterio_prob and criterio_quota:
                # Calcolo stake
                kelly_fraction = edge_value / (quota_over - 1)
                
                if edge_value >= 0.15 and probabilita_finale >= 0.65:
                    stake_raw = bankroll * kelly_fraction * 0.33
                    stake = min(stake_raw, bankroll * 0.05)
                    confidenza = "Alta"
                else:
                    stake_raw = bankroll * kelly_fraction * 0.25
                    stake = min(stake_raw, bankroll * 0.025)
                    confidenza = "Media"
                
                stake = max(1.00, round(stake, 2))
                
                scommessa = {
                    'line': line,
                    'quota_over': quota_over,
                    'prob_finale': probabilita_finale,
                    'edge': edge_value,
                    'stake': stake,
                    'confidenza': confidenza
                }
                
                scommesse_valide.append(scommessa)
                st.success(f"✅ SCOMMESSA VALIDA - Stake: {stake}€")
            else:
                st.warning("❌ Non soddisfa criteri")
                if not criterio_edge:
                    st.write(f"  - Edge insufficiente ({edge_value*100:.1f}% < 10%)")
                if not criterio_prob:
                    st.write(f"  - Probabilità bassa ({probabilita_finale*100:.1f}% < 60%)")
                if not criterio_quota:
                    st.write(f"  - Quota fuori range ({quota_over} non in 1.70-1.80)")
    
    return scommesse_valide


# Funzioni di supporto
def prepare_features_from_team_stats_streamlit(team_stats: Dict) -> Dict:
    """Prepara features da team_stats per ML"""
    home = team_stats['home']
    away = team_stats['away']
    
    features = {
        'HOME_ORtg_sAvg': home.get('ORtg_season', 110.0),
        'HOME_DRtg_sAvg': home.get('DRtg_season', 110.0),
        'HOME_PACE': home.get('Pace_season', 100.0),
        'AWAY_ORtg_sAvg': away.get('ORtg_season', 110.0),
        'AWAY_DRtg_sAvg': away.get('DRtg_season', 110.0),
        'AWAY_PACE': away.get('Pace_season', 100.0),
        'HOME_eFG_PCT_sAvg': home.get('eFG_season', 0.52),
        'HOME_TOV_PCT_sAvg': home.get('TOV_season', 0.14),
        'HOME_OREB_PCT_sAvg': home.get('OREB_season', 0.25),
        'HOME_FT_RATE_sAvg': home.get('FT_rate_season', 0.22),
        'AWAY_eFG_PCT_sAvg': away.get('eFG_season', 0.52),
        'AWAY_TOV_PCT_sAvg': away.get('TOV_season', 0.14),
        'AWAY_OREB_PCT_sAvg': away.get('OREB_season', 0.25),
        'AWAY_FT_RATE_sAvg': away.get('FT_rate_season', 0.22),
        'HOME_ORtg_L5Avg': home.get('ORtg_L5', home.get('ORtg_season', 110.0)),
        'HOME_DRtg_L5Avg': home.get('DRtg_L5', home.get('DRtg_season', 110.0)),
        'AWAY_ORtg_L5Avg': away.get('ORtg_L5', away.get('ORtg_season', 110.0)),
        'AWAY_DRtg_L5Avg': away.get('DRtg_L5', away.get('DRtg_season', 110.0))
    }
    
    # Features derivate
    features['GAME_PACE'] = (features['HOME_PACE'] + features['AWAY_PACE']) / 2
    
    return features


def dict_to_feature_vector_streamlit(features_dict: Dict, feature_names: List[str]) -> List[float]:
    """Converte dict features in vettore per ML"""
    return [features_dict.get(name, 0.0) for name in feature_names]


def predict_complete_game_streamlit(game_data: Dict) -> Optional[Dict]:
    """
    Pipeline completo NBA per Streamlit
    Versione ottimizzata con progress tracking
    """
    
    # Progress bar principale
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Livello 1
        status_text.text("🎯 Esecuzione Livello 1...")
        progress_bar.progress(25)
        
        livello1_results = execute_livello1_streamlit(game_data)
        media_punti_l1 = livello1_results['Media_punti_stimati_L1_finale']
        
        # Livello 2
        status_text.text("📊 Esecuzione Livello 2...")
        progress_bar.progress(50)
        
        sd_results = calculate_sd_final_streamlit(game_data, livello1_results)
        sd_final = sd_results['sd_final']
        
        # Monte Carlo
        quote_valide = [q for q in game_data['quotes'] if 1.70 <= q.get('over_quote', 0) <= 1.95]
        monte_carlo_results = execute_monte_carlo_streamlit(media_punti_l1, sd_final, quote_valide)
        
        # Livello 3
        status_text.text("🤖 Esecuzione Livello 3...")
        progress_bar.progress(75)
        
        livello3_results = execute_livello3_ml_streamlit(media_punti_l1, game_data['team_stats'])
        
        # Value Bet Analysis
        status_text.text("💡 Analisi Value Betting...")
        progress_bar.progress(90)
        
        bankroll = game_data.get('settings', {}).get('bankroll', 100.0)
        scommesse_valide = execute_value_bet_analysis_streamlit(
            monte_carlo_results, livello3_results, quote_valide, bankroll
        )
        
        status_text.text("✅ Predizione completata!")
        progress_bar.progress(100)
        
        return {
            'livello1': livello1_results,
            'livello2': {'sd_results': sd_results, 'monte_carlo': monte_carlo_results},
            'livello3': livello3_results,
            'recommendations': scommesse_valide,
            'media_punti': media_punti_l1,
            'sd_final': sd_final
        }
        
    except Exception as e:
        status_text.text(f"❌ Errore: {e}")
        st.error(f"Errore durante la predizione: {e}")
        return None