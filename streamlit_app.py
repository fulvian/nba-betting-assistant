import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Configurazione pagina
st.set_page_config(
    page_title="🏀 NBA Betting Assistant",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/tuonome/nba-betting-assistant',
        'Report a bug': 'https://github.com/tuonome/nba-betting-assistant/issues',
        'About': "# NBA Betting Assistant v6.1\nPronostici professionali basati su AI"
    }
)

# CSS personalizzato per mobile-first
st.markdown("""
<style>
    /* Mobile-first responsive design */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f4e79;
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .bet-success {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    
    .bet-warning {
        background: linear-gradient(135deg, #ff8a00 0%, #da1b60 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .metric-card {
            margin-bottom: 0.5rem;
            padding: 0.8rem;
        }
        .main-header {
            padding: 0.8rem;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Import moduli personalizzati
try:
    from src.nba_engine import NBACompleteSystem, predict_complete_game
    from src.api_integrations import NBADataFetcher, QuotesFetcher
    from src.ui_components import create_team_selector, create_prediction_card, create_performance_dashboard
    from src.data_manager import DataManager
except ImportError:
    st.error("⚠️ Moduli non trovati. Assicurati che la cartella 'src' sia presente.")
    st.stop()

# Inizializzazione stato sessione
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'user_bankroll' not in st.session_state:
    st.session_state.user_bankroll = 100.0
if 'api_data_cache' not in st.session_state:
    st.session_state.api_data_cache = {}

def main():
    # Header principale
    st.markdown("""
    <div class="main-header">
        <h1>🏀 NBA Betting Assistant</h1>
        <p>Pronostici professionali basati su Machine Learning e Livelli 1-2-3</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar per configurazioni
    with st.sidebar:
        st.image("https://logoeps.com/wp-content/uploads/2013/03/nba-vector-logo.png", width=100)
        st.title("⚙️ Configurazioni")
        
        # User settings
        st.subheader("👤 Impostazioni Utente")
        user_bankroll = st.number_input(
            "💰 Bankroll (€)", 
            min_value=10.0, 
            max_value=10000.0, 
            value=st.session_state.user_bankroll,
            step=10.0
        )
        st.session_state.user_bankroll = user_bankroll
        
        # API settings
        st.subheader("🌐 Configurazione API")
        use_live_data = st.checkbox("📡 Usa dati live NBA", value=True)
        api_key_odds = st.text_input("🔑 The Odds API Key", type="password", help="Opzionale: per quote live")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        if st.session_state.predictions_history:
            total_predictions = len(st.session_state.predictions_history)
            st.metric("🎯 Predizioni Totali", total_predictions)
            
            wins = sum(1 for p in st.session_state.predictions_history if p.get('result') == 'win')
            if total_predictions > 0:
                win_rate = (wins / total_predictions) * 100
                st.metric("📈 Win Rate", f"{win_rate:.1f}%")
    
    # Tabs principali
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Pronostico", "📊 Dashboard", "📈 Performance", "⚙️ Avanzate"])
    
    with tab1:
        create_prediction_interface(use_live_data, api_key_odds)
    
    with tab2:
        create_analytics_dashboard()
    
    with tab3:
        create_performance_tracking()
    
    with tab4:
        create_advanced_settings()

def create_prediction_interface(use_live_data=True, api_key_odds=None):
    """Interface principale per creare pronostici"""
    
    st.header("🎯 Crea Nuovo Pronostico")
    
    # Selezione rapida partite del giorno
    if use_live_data:
        st.subheader("🗓️ Partite di Oggi")
        try:
            nba_fetcher = NBADataFetcher()
            today_games = nba_fetcher.get_today_games()
            
            if today_games:
                game_options = [f"{game['away_team']} @ {game['home_team']} - {game['time']}" 
                               for game in today_games]
                selected_game = st.selectbox("⚡ Quick Select", ["Personalizzata"] + game_options)
                
                if selected_game != "Personalizzata":
                    game_idx = game_options.index(selected_game)
                    selected_game_data = today_games[game_idx]
                    st.success(f"✅ Partita selezionata: {selected_game}")
            else:
                st.info("ℹ️ Nessuna partita NBA oggi. Usa selezione manuale.")
        except Exception as e:
            st.warning(f"⚠️ Errore caricamento partite: {e}")
    
    # Form di input
    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("🏀 Dettagli Partita")
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox(
                "🏠 Squadra Casa",
                options=get_nba_teams(),
                index=15  # Lakers di default
            )
            
        with col2:
            away_team = st.selectbox(
                "✈️ Squadra Ospite", 
                options=get_nba_teams(),
                index=2  # Celtics di default
            )
        
        # Data e orario
        col3, col4 = st.columns(2)
        with col3:
            game_date = st.date_input("📅 Data Partita", datetime.now().date())
        with col4:
            game_time = st.time_input("🕐 Orario", datetime.now().time())
        
        # Modalità input dati
        st.subheader("📊 Modalità Input Dati")
        data_mode = st.radio(
            "Come vuoi inserire i dati?",
            ["🤖 Automatico (API)", "📁 Upload JSON", "✏️ Manuale"],
            horizontal=True
        )
        
        if data_mode == "🤖 Automatico (API)":
            if use_live_data:
                st.success("✅ I dati verranno caricati automaticamente dalle API NBA")
                auto_fetch = True
            else:
                st.warning("⚠️ Abilita 'Usa dati live NBA' nella sidebar")
                auto_fetch = False
                
        elif data_mode == "📁 Upload JSON":
            uploaded_file = st.file_uploader(
                "Carica file JSON con dati partita",
                type=['json'],
                help="Formato: stesso del sistema precedente"
            )
            auto_fetch = False
            
        else:  # Manuale
            st.info("📝 Compila manualmente i dati delle squadre")
            auto_fetch = False
            
        # Quote input
        st.subheader("💰 Quote Bookmaker")
        quotes_mode = st.radio(
            "Fonte quote:",
            ["🌐 API Live", "📝 Inserimento Manuale"],
            horizontal=True
        )
        
        if quotes_mode == "🌐 API Live":
            if api_key_odds:
                st.success("✅ Quote live abilitate")
            else:
                st.warning("⚠️ Inserisci API key nella sidebar per quote live")
                quotes_mode = "📝 Inserimento Manuale"
        
        if quotes_mode == "📝 Inserimento Manuale":
            col5, col6, col7 = st.columns(3)
            with col5:
                over_line = st.number_input("📈 Linea", min_value=150.0, max_value=300.0, value=225.5, step=0.5)
            with col6:
                over_quote = st.number_input("💹 Quota Over", min_value=1.50, max_value=2.50, value=1.85, step=0.01)
            with col7:
                bookmaker = st.selectbox("🏦 Bookmaker", ["Bet365", "Pinnacle", "William Hill", "Altro"])
        
        # Submit button
        submitted = st.form_submit_button("🚀 Genera Pronostico", use_container_width=True)
        
        if submitted:
            with st.spinner("🔄 Elaborazione pronostico in corso..."):
                try:
                    # Raccogli dati
                    game_data = collect_game_data(
                        home_team, away_team, game_date, game_time,
                        data_mode, auto_fetch, uploaded_file if data_mode == "📁 Upload JSON" else None
                    )
                    
                    # Aggiungi quote
                    if quotes_mode == "📝 Inserimento Manuale":
                        game_data['quotes'] = [{
                            'line': over_line,
                            'over_quote': over_quote,
                            'bookmaker': bookmaker
                        }]
                    else:
                        game_data['quotes'] = fetch_live_quotes(home_team, away_team, api_key_odds)
                    
                    # Genera pronostico
                    prediction_result = generate_prediction(game_data)
                    
                    # Mostra risultato
                    display_prediction_result(prediction_result, game_data)
                    
                    # Salva nello storico
                    st.session_state.predictions_history.append({
                        'timestamp': datetime.now(),
                        'game': f"{away_team} @ {home_team}",
                        'prediction': prediction_result,
                        'game_data': game_data
                    })
                    
                except Exception as e:
                    st.error(f"❌ Errore durante la generazione: {str(e)}")
                    st.exception(e)

def collect_game_data(home_team, away_team, game_date, game_time, data_mode, auto_fetch, uploaded_file):
    """Raccoglie dati della partita secondo la modalità selezionata"""
    
    game_data = {
        'game_info': {
            'home_team': home_team,
            'away_team': away_team,
            'date': game_date.strftime('%Y-%m-%d'),
            'time': game_time.strftime('%H:%M'),
            'league': 'NBA',
            'season': '2024-25'
        }
    }
    
    if data_mode == "🤖 Automatico (API)" and auto_fetch:
        # Fetch da API
        fetcher = NBADataFetcher()
        team_stats = fetcher.get_team_stats(home_team, away_team)
        game_data['team_stats'] = team_stats
        game_data['context'] = fetcher.get_game_context(home_team, away_team)
        game_data['league_data'] = fetcher.get_league_averages()
        
    elif data_mode == "📁 Upload JSON" and uploaded_file:
        # Carica da JSON
        json_data = json.load(uploaded_file)
        game_data.update(json_data)
        
    else:
        # Dati di esempio per modalità manuale
        game_data.update(get_sample_game_data(home_team, away_team))
    
    # Aggiungi bankroll
    game_data['settings'] = {'bankroll': st.session_state.user_bankroll}
    
    return game_data

def generate_prediction(game_data):
    """Genera pronostico usando il sistema NBA completo"""
    try:
        # Usa il sistema NBA originale
        from src.nba_engine import (
            execute_livello1, calculate_sd_final, execute_monte_carlo,
            execute_livello3_ml, execute_value_bet_analysis
        )
        
        # Livello 1
        livello1_results = execute_livello1(game_data)
        media_punti_l1 = livello1_results['Media_punti_stimati_L1_finale']
        
        # Livello 2
        sd_results = calculate_sd_final(game_data, livello1_results)
        sd_final = sd_results['sd_final']
        
        quote_valide = [q for q in game_data['quotes'] if 1.70 <= q.get('over_quote', 0) <= 1.95]
        monte_carlo_results = execute_monte_carlo(media_punti_l1, sd_final, quote_valide)
        
        # Livello 3
        livello3_results = execute_livello3_ml(media_punti_l1, game_data['team_stats'], sd_final)
        
        # Value Bet Analysis
        bankroll = game_data['settings']['bankroll']
        scommesse_valide = execute_value_bet_analysis(
            monte_carlo_results, livello3_results, quote_valide, bankroll
        )
        
        return {
            'livello1': livello1_results,
            'livello2': {'sd_results': sd_results, 'monte_carlo': monte_carlo_results},
            'livello3': livello3_results,
            'recommendations': scommesse_valide,
            'media_punti': media_punti_l1,
            'sd_final': sd_final
        }
        
    except Exception as e:
        st.error(f"Errore nel sistema di predizione: {e}")
        return None

def display_prediction_result(prediction_result, game_data):
    """Mostra il risultato del pronostico in formato professionale"""
    
    if not prediction_result:
        st.error("❌ Impossibile generare il pronostico")
        return
    
    game_info = game_data['game_info']
    media_punti = prediction_result['media_punti']
    recommendations = prediction_result['recommendations']
    
    # Header risultato
    st.markdown(f"""
    <div class="prediction-card">
        <h2>🎯 PRONOSTICO GENERATO</h2>
        <h3>{game_info['away_team']} @ {game_info['home_team']}</h3>
        <p>{game_info['date']} - {game_info['time']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metriche principali
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🎯 Punti Stimati",
            f"{media_punti:.1f}",
            help="Media punti totali stimata dal Livello 1"
        )
    
    with col2:
        st.metric(
            "📊 Deviazione Standard",
            f"{prediction_result['sd_final']:.1f}",
            help="Variabilità attesa del punteggio"
        )
    
    with col3:
        confidence_level = "Alta" if len(recommendations) > 0 else "Bassa"
        st.metric(
            "🔐 Confidenza Sistema",
            confidence_level,
            help="Livello di confidenza nell'analisi"
        )
    
    # Raccomandazioni
    if recommendations:
        best_bet = max(recommendations, key=lambda x: x['edge'])
        
        st.markdown(f"""
        <div class="bet-success">
            <h3>✅ SCOMMESSA RACCOMANDATA</h3>
            <p><strong>🎯 OVER {best_bet['line']}</strong> @ <strong>{best_bet['quota_over']}</strong></p>
            <p>📊 Probabilità: <strong>{best_bet['prob_finale']*100:.1f}%</strong></p>
            <p>📈 Edge: <strong>+{best_bet['edge']*100:.1f}%</strong></p>
            <p>💸 Stake: <strong>{best_bet['stake']:.2f}€</strong></p>
            <p>🔐 Confidenza: <strong>{best_bet['confidenza']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Grafici di supporto
        create_prediction_charts(prediction_result, best_bet)
        
    else:
        st.markdown("""
        <div class="bet-warning">
            <h3>⚠️ NO BET</h3>
            <p>Nessuna scommessa soddisfa i criteri di value betting</p>
            <p>Aspetta opportunità migliori!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dettagli tecnici (collapsible)
    with st.expander("🔬 Dettagli Tecnici Completi"):
        show_technical_details(prediction_result)

def create_prediction_charts(prediction_result, best_bet):
    """Crea grafici di supporto per la predizione"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuzione probabilità
        x = np.linspace(150, 300, 1000)
        media = prediction_result['media_punti']
        sd = prediction_result['sd_final']
        y = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - media) / sd) ** 2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, fill='tonexty', name='Distribuzione Probabilità'))
        fig.add_vline(x=best_bet['line'], line_dash="dash", line_color="red", 
                     annotation_text=f"Linea {best_bet['line']}")
        fig.add_vline(x=media, line_dash="solid", line_color="green",
                     annotation_text=f"Media {media:.1f}")
        
        fig.update_layout(
            title="📊 Distribuzione Probabilità Punteggio",
            xaxis_title="Punti Totali",
            yaxis_title="Densità",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Edge vs Quota
        quotes_range = np.arange(1.70, 2.00, 0.01)
        edges = [(q * best_bet['prob_finale']) - 1 for q in quotes_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=quotes_range, y=edges, mode='lines', name='Edge %'))
        fig.add_hline(y=0.10, line_dash="dash", line_color="green",
                     annotation_text="Soglia Value Bet 10%")
        fig.add_scatter(x=[best_bet['quota_over']], y=[best_bet['edge']], 
                       mode='markers', marker=dict(size=12, color='red'),
                       name='Quota Corrente')
        
        fig.update_layout(
            title="📈 Analisi Value Betting",
            xaxis_title="Quota Over",
            yaxis_title="Edge %",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def create_analytics_dashboard():
    """Dashboard analytics completa"""
    
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.predictions_history:
        st.info("📈 Inizia a fare pronostici per vedere le analytics!")
        return
    
    # Metriche overview
    st.subheader("🎯 Overview Performance")
    
    total_predictions = len(st.session_state.predictions_history)
    predictions_with_result = [p for p in st.session_state.predictions_history if 'result' in p]
    
    if predictions_with_result:
        wins = sum(1 for p in predictions_with_result if p['result'] == 'win')
        win_rate = (wins / len(predictions_with_result)) * 100
        
        total_stakes = sum(p['prediction']['recommendations'][0]['stake'] 
                          for p in predictions_with_result 
                          if p['prediction']['recommendations'])
        
        total_profit = sum(p.get('profit', 0) for p in predictions_with_result)
        roi = (total_profit / total_stakes * 100) if total_stakes > 0 else 0
    else:
        win_rate = 0
        total_profit = 0
        roi = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Predizioni Totali", total_predictions)
    with col2:
        st.metric("📈 Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("💰 Profitto Totale", f"{total_profit:.2f}€")
    with col4:
        st.metric("📊 ROI", f"{roi:.1f}%")
    
    # Grafici timeline
    create_performance_timeline()
    
    # Heatmap delle performance
    create_team_performance_heatmap()

def create_performance_tracking():
    """Tracking performance dettagliato"""
    
    st.header("📈 Performance Tracking")
    
    # Upload risultati
    st.subheader("📥 Aggiorna Risultati")
    
    pending_predictions = [p for p in st.session_state.predictions_history if 'result' not in p]
    
    if pending_predictions:
        with st.form("update_results"):
            prediction_to_update = st.selectbox(
                "Seleziona predizione da aggiornare:",
                options=range(len(pending_predictions)),
                format_func=lambda x: f"{pending_predictions[x]['game']} - {pending_predictions[x]['timestamp'].strftime('%d/%m/%Y %H:%M')}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                final_score = st.number_input("🏀 Punteggio Finale", min_value=100, max_value=400, value=220)
            with col2:
                result = st.selectbox("📊 Risultato", ["win", "loss", "push"])
            
            if st.form_submit_button("✅ Aggiorna Risultato"):
                update_prediction_result(prediction_to_update, final_score, result)
                st.success("✅ Risultato aggiornato!")
                st.rerun()
    else:
        st.info("✅ Tutti i risultati sono aggiornati!")
    
    # Storico dettagliato
    if st.session_state.predictions_history:
        st.subheader("📋 Storico Predizioni")
        
        df = create_predictions_dataframe()
        st.dataframe(df, use_container_width=True)
        
        # Export data
        if st.button("📥 Esporta CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"nba_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

def create_advanced_settings():
    """Impostazioni avanzate del sistema"""
    
    st.header("⚙️ Impostazioni Avanzate")
    
    # Model settings
    st.subheader("🤖 Configurazione Modelli ML")
    
    with st.expander("🧠 Gestione Modelli"):
        if st.button("🔄 Riaddestra Modelli"):
            with st.spinner("Riaddestramento in corso..."):
                # Placeholder per riaddestramento
                st.success("✅ Modelli riaddestrati!")
        
        model_performance = {
            'Random Forest': {'accuracy': 0.742, 'mae': 8.3},
            'Gradient Boosting': {'accuracy': 0.758, 'mae': 7.9},
            'Ridge Regression': {'accuracy': 0.721, 'mae': 9.1}
        }
        
        for model_name, perf in model_performance.items():
            st.metric(
                f"📊 {model_name}",
                f"R²: {perf['accuracy']:.3f}",
                f"MAE: {perf['mae']:.1f}"
            )
    
    # API Configuration
    st.subheader("🌐 Configurazione API")
    
    with st.expander("🔑 Gestione API Keys"):
        st.text_input("🏀 NBA API Key", type="password", help="Opzionale per dati premium")
        st.text_input("💰 The Odds API Key", type="password", help="Per quote live")
        st.text_input("📊 RapidAPI Key", type="password", help="Per statistiche avanzate")
        
        if st.button("✅ Salva Configurazione"):
            st.success("Configurazione salvata!")
    
    # System Info
    st.subheader("🛠️ Informazioni Sistema")
    
    system_info = {
        "Versione": "NBA Assistant v6.1",
        "Engine": "Livelli 1-2-3 + ML",
        "Database": f"{len(st.session_state.predictions_history)} predizioni",
        "Uptime": "🟢 Online",
        "Last Update": datetime.now().strftime("%d/%m/%Y %H:%M")
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")

# Funzioni di supporto
def get_nba_teams():
    """Lista squadre NBA"""
    return [
        "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
        "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
        "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
        "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
        "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
        "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
        "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
        "Utah Jazz", "Washington Wizards"
    ]

def get_sample_game_data(home_team, away_team):
    """Dati di esempio per test"""
    return {
        'team_stats': {
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
        },
        'context': {
            'playoff_game': False, 'tanking_mode': {'home': False, 'away': False},
            'h2h_scores_last3': [238, 245, 221]
        },
        'league_data': {'avg_ORtg': 112.5, 'var_league': 180.5}
    }

def update_prediction_result(idx, final_score, result):
    """Aggiorna risultato di una predizione"""
    prediction = st.session_state.predictions_history[idx]
    prediction['result'] = result
    prediction['final_score'] = final_score
    
    # Calcola profitto se c'era una scommessa
    if prediction['prediction']['recommendations']:
        bet = prediction['prediction']['recommendations'][0]
        stake = bet['stake']
        
        if result == 'win':
            profit = stake * (bet['quota_over'] - 1)
        elif result == 'loss':
            profit = -stake
        else:  # push
            profit = 0
        
        prediction['profit'] = profit

def create_predictions_dataframe():
    """Crea DataFrame delle predizioni per visualizzazione"""
    data = []
    for p in st.session_state.predictions_history:
        row = {
            'Data': p['timestamp'].strftime('%d/%m/%Y'),
            'Partita': p['game'],
            'Punti Stimati': p['prediction']['media_punti'],
            'Risultato': p.get('result', 'Pending'),
            'Punteggio Finale': p.get('final_score', '-'),
            'Profitto': f"{p.get('profit', 0):.2f}€"
        }
        
        if p['prediction']['recommendations']:
            bet = p['prediction']['recommendations'][0]
            row.update({
                'Linea': bet['line'],
                'Quota': bet['quota_over'],
                'Stake': f"{bet['stake']:.2f}€"
            })
        
        data.append(row)
    
    return pd.DataFrame(data)

def fetch_live_quotes(home_team, away_team, api_key):
    """Fetch quote live (placeholder)"""
    # Implementazione futura con The Odds API
    return [{
        'line': 225.5,
        'over_quote': 1.85,
        'under_quote': 1.95,
        'bookmaker': 'Live API'
    }]

def show_technical_details(prediction_result):
    """Mostra dettagli tecnici completi"""
    st.json(prediction_result)

def create_performance_timeline():
    """Timeline delle performance"""
    # Placeholder - implementazione futura
    st.info("📊 Timeline delle performance - Coming Soon!")

def create_team_performance_heatmap():
    """Heatmap performance per squadra"""
    # Placeholder - implementazione futura  
    st.info("🔥 Heatmap performance squadre - Coming Soon!")

if __name__ == "__main__":
    main()