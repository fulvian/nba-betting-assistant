import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

def create_team_selector(key_suffix: str = "") -> Tuple[str, str]:
    """
    Crea selettore squadre NBA avanzato con ricerca
    """
    teams = get_nba_teams_with_logos()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Squadra Casa")
        home_team = st.selectbox(
            "Seleziona squadra casa:",
            options=list(teams.keys()),
            key=f"home_team_{key_suffix}",
            format_func=lambda x: f"{teams[x]['emoji']} {x}"
        )
        
        if home_team:
            st.image(teams[home_team]['logo'], width=80)
            st.caption(f"Conference: {teams[home_team]['conference']}")
    
    with col2:
        st.subheader("✈️ Squadra Ospite")
        away_options = [team for team in teams.keys() if team != home_team]
        away_team = st.selectbox(
            "Seleziona squadra ospite:",
            options=away_options,
            key=f"away_team_{key_suffix}",
            format_func=lambda x: f"{teams[x]['emoji']} {x}"
        )
        
        if away_team:
            st.image(teams[away_team]['logo'], width=80)
            st.caption(f"Conference: {teams[away_team]['conference']}")
    
    return home_team, away_team

def create_prediction_card(prediction_data: Dict, game_info: Dict) -> None:
    """
    Crea card predizione professionale con animazioni
    """
    # Gradient background based on confidence
    recommendations = prediction_data.get('recommendations', [])
    
    if recommendations:
        best_bet = max(recommendations, key=lambda x: x.get('edge', 0))
        confidence_color = get_confidence_color(best_bet['confidenza'])
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {confidence_color['primary']}, {confidence_color['secondary']});
            padding: 2rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        ">
            <h2>🎯 PREDIZIONE NBA</h2>
            <h3>{game_info['away_team']} @ {game_info['home_team']}</h3>
            <p style="font-size: 1.1em; margin: 1rem 0;">
                📅 {game_info['date']} • 🕐 {game_info.get('time', 'TBD')}
            </p>
            
            <div style="
                background: rgba(255,255,255,0.2);
                border-radius: 15px;
                padding: 1.5rem;
                margin: 1rem 0;
            ">
                <h1 style="margin: 0; font-size: 2.5em;">OVER {best_bet['line']}</h1>
                <p style="font-size: 1.2em; margin: 0.5rem 0;">@ {best_bet['quota_over']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metriche in card separate
        create_metrics_row([
            ("🎯 Punti Stimati", f"{prediction_data['media_punti']:.1f}"),
            ("📊 Probabilità", f"{best_bet['prob_finale']*100:.1f}%"),
            ("📈 Edge", f"+{best_bet['edge']*100:.1f}%"),
            ("💸 Stake", f"{best_bet['stake']:.2f}€"),
            ("🔐 Confidenza", best_bet['confidenza'])
        ])
        
    else:
        # NO BET card
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff6b6b, #feca57);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin: 1rem 0;
        ">
            <h2>⚠️ NO BET</h2>
            <h3>{} @ {}</h3>
            <p>Nessuna opportunità di value betting identificata</p>
            <p>🎯 Punti Stimati: {:.1f}</p>
        </div>
        """.format(
            game_info['away_team'], 
            game_info['home_team'],
            prediction_data['media_punti']
        ), unsafe_allow_html=True)

def create_metrics_row(metrics: List[Tuple[str, str]]) -> None:
    """
    Crea riga di metriche responsive
    """
    num_metrics = len(metrics)
    cols = st.columns(num_metrics)
    
    for i, (label, value) in enumerate(metrics):
        with cols[i]:
            st.metric(label, value)

def create_performance_dashboard(predictions_history: List[Dict]) -> None:
    """
    Dashboard performance completa con grafici interattivi
    """
    if not predictions_history:
        st.info("📊 Inizia a fare predizioni per vedere le performance!")
        return
    
    # Calcola metriche
    completed_predictions = [p for p in predictions_history if 'result' in p]
    
    if not completed_predictions:
        st.warning("⏳ Nessuna predizione completata ancora")
        return
    
    # Overview metrics
    total_bets = len(completed_predictions)
    wins = sum(1 for p in completed_predictions if p['result'] == 'win')
    total_profit = sum(p.get('profit', 0) for p in completed_predictions)
    
    win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
    
    # Layout principale
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Predizioni", total_bets)
    with col2:
        st.metric("📈 Win Rate", f"{win_rate:.1f}%", 
                 delta=f"{win_rate-52:.1f}%" if win_rate > 52 else None)
    with col3:
        st.metric("💰 Profitto", f"{total_profit:.2f}€",
                 delta=f"+{total_profit:.2f}€" if total_profit > 0 else None)
    with col4:
        avg_stake = np.mean([p.get('stake', 0) for p in completed_predictions])
        roi = (total_profit / (avg_stake * total_bets)) * 100 if avg_stake > 0 else 0
        st.metric("📊 ROI", f"{roi:.1f}%")
    
    # Grafici
    create_performance_charts(completed_predictions)

def create_performance_charts(predictions: List[Dict]) -> None:
    """
    Crea grafici performance dettagliati
    """
    # Prepare data
    df = pd.DataFrame([
        {
            'date': p['timestamp'].date(),
            'result': p['result'],
            'profit': p.get('profit', 0),
            'cumulative_profit': 0,  # Calcolato dopo
            'game': p['game']
        }
        for p in predictions
    ])
    
    # Calcola profitto cumulativo
    df = df.sort_values('date')
    df['cumulative_profit'] = df['profit'].cumsum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Grafico profitto cumulativo
        fig = px.line(df, x='date', y='cumulative_profit',
                     title='💰 Profitto Cumulativo nel Tempo',
                     markers=True)
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                     annotation_text="Break Even")
        
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Profitto (€)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribuzione risultati
        result_counts = df['result'].value_counts()
        
        fig = px.pie(values=result_counts.values, 
                    names=result_counts.index,
                    title='📊 Distribuzione Risultati',
                    color_discrete_map={
                        'win': '#2ecc71',
                        'loss': '#e74c3c', 
                        'push': '#f39c12'
                    })
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap performance per giorno della settimana
    create_weekday_heatmap(df)

def create_weekday_heatmap(df: pd.DataFrame) -> None:
    """
    Crea heatmap performance per giorno della settimana
    """
    df['weekday'] = pd.to_datetime(df['date']).dt.day_name()
    df['week'] = pd.to_datetime(df['date']).dt.isocalendar().week
    
    # Aggrega per settimana e giorno
    heatmap_data = df.groupby(['week', 'weekday'])['profit'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='week', columns='weekday', values='profit')
    
    # Riordina giorni
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex(columns=day_order, fill_value=0)
    
    fig = px.imshow(heatmap_pivot.values,
                   x=heatmap_pivot.columns,
                   y=heatmap_pivot.index,
                   title='🔥 Heatmap Performance per Giorno',
                   color_continuous_scale='RdYlGn',
                   aspect='auto')
    
    fig.update_layout(
        xaxis_title="Giorno della Settimana",
        yaxis_title="Settimana",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_advanced_filters() -> Dict:
    """
    Crea filtri avanzati per analisi
    """
    st.subheader("🔍 Filtri Avanzati")
    
    with st.expander("⚙️ Configurazioni Analisi"):
        col1, col2 = st.columns(2)
        
        with col1:
            date_range = st.date_input(
                "📅 Periodo Analisi",
                value=(datetime.now().date() - timedelta(days=30), datetime.now().date()),
                max_value=datetime.now().date()
            )
            
            min_confidence = st.selectbox(
                "🔐 Confidenza Minima",
                options=['Tutte', 'Bassa', 'Media', 'Alta'],
                index=0
            )
        
        with col2:
            team_filter = st.multiselect(
                "🏀 Filtra per Squadre",
                options=list(get_nba_teams_with_logos().keys()),
                default=[]
            )
            
            bet_type = st.selectbox(
                "🎯 Tipo Scommessa",
                options=['Tutte', 'Solo Value Bet', 'Solo No Bet'],
                index=0
            )
    
    return {
        'date_range': date_range,
        'min_confidence': min_confidence,
        'team_filter': team_filter,
        'bet_type': bet_type
    }

def create_comparison_chart(team1_stats: Dict, team2_stats: Dict, 
                          team1_name: str, team2_name: str) -> None:
    """
    Crea grafico comparativo tra due squadre
    """
    categories = ['ORtg', 'DRtg', 'Pace', 'eFG%', 'TOV%', 'OREB%']
    
    team1_values = [
        team1_stats.get('ORtg_season', 110),
        120 - team1_stats.get('DRtg_season', 110),  # Inverted for better visual
        team1_stats.get('Pace_season', 100),
        team1_stats.get('eFG_season', 0.5) * 100,
        (1 - team1_stats.get('TOV_season', 0.15)) * 100,
        team1_stats.get('OREB_season', 0.25) * 100
    ]
    
    team2_values = [
        team2_stats.get('ORtg_season', 110),
        120 - team2_stats.get('DRtg_season', 110),
        team2_stats.get('Pace_season', 100),
        team2_stats.get('eFG_season', 0.5) * 100,
        (1 - team2_stats.get('TOV_season', 0.15)) * 100,
        team2_stats.get('OREB_season', 0.25) * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=team1_values,
        theta=categories,
        fill='toself',
        name=team1_name,
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=team2_values,
        theta=categories,
        fill='toself',
        name=team2_name,
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 130]
            )),
        title="📊 Confronto Statistiche Squadre",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_bet_recommendation_card(recommendation: Dict) -> None:
    """
    Card raccomandazione scommessa con styling avanzato
    """
    confidence_colors = get_confidence_color(recommendation['confidenza'])
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {confidence_colors['primary']}, {confidence_colors['secondary']});
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-left: 5px solid #fff;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h3 style="margin: 0; font-size: 1.5em;">🎯 OVER {recommendation['line']}</h3>
                <p style="margin: 0.5rem 0; font-size: 1.1em;">@ {recommendation['quota_over']}</p>
            </div>
            <div style="text-align: right;">
                <h2 style="margin: 0; font-size: 2em;">{recommendation['stake']:.2f}€</h2>
                <p style="margin: 0; opacity: 0.9;">Stake consigliato</p>
            </div>
        </div>
        
        <div style="
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
        ">
            <div style="text-align: center;">
                <strong>{recommendation['prob_finale']*100:.1f}%</strong>
                <br><small>Probabilità</small>
            </div>
            <div style="text-align: center;">
                <strong>+{recommendation['edge']*100:.1f}%</strong>
                <br><small>Edge</small>
            </div>
            <div style="text-align: center;">
                <strong>{recommendation['confidenza']}</strong>
                <br><small>Confidenza</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_loading_animation(text: str = "Elaborazione in corso...") -> None:
    """
    Animazione di caricamento personalizzata
    """
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 2rem;
        font-size: 1.2em;
        color: #1f4e79;
    ">
        <div style="
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #1f4e79;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 1rem;
        "></div>
        <br>
        {text}
    </div>
    
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """, unsafe_allow_html=True)

# Funzioni di supporto
def get_confidence_color(confidence: str) -> Dict[str, str]:
    """
    Restituisce colori per livello di confidenza
    """
    colors = {
        'Alta': {'primary': '#27ae60', 'secondary': '#2ecc71'},
        'Media': {'primary': '#f39c12', 'secondary': '#e67e22'},
        'Bassa': {'primary': '#e74c3c', 'secondary': '#c0392b'}
    }
    return colors.get(confidence, colors['Media'])

def get_nba_teams_with_logos() -> Dict[str, Dict]:
    """
    Restituisce dictionary completo squadre NBA con loghi ed emoji
    """
    return {
        "Atlanta Hawks": {
            "emoji": "🦅",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/atlanta-hawks-vector-logo.png"
        },
        "Boston Celtics": {
            "emoji": "🍀",
            "conference": "East", 
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/boston-celtics-vector-logo.png"
        },
        "Brooklyn Nets": {
            "emoji": "🕸️",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/brooklyn-nets-vector-logo.png"
        },
        "Charlotte Hornets": {
            "emoji": "🐝",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/charlotte-hornets-vector-logo.png"
        },
        "Chicago Bulls": {
            "emoji": "🐂",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/chicago-bulls-vector-logo.png"
        },
        "Cleveland Cavaliers": {
            "emoji": "⚔️",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/cleveland-cavaliers-vector-logo.png"
        },
        "Dallas Mavericks": {
            "emoji": "🐎",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/dallas-mavericks-vector-logo.png"
        },
        "Denver Nuggets": {
            "emoji": "🏔️",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/denver-nuggets-vector-logo.png"
        },
        "Detroit Pistons": {
            "emoji": "🔧",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/detroit-pistons-vector-logo.png"
        },
        "Golden State Warriors": {
            "emoji": "🌉",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/golden-state-warriors-vector-logo.png"
        },
        "Houston Rockets": {
            "emoji": "🚀",
            "conference": "West", 
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/houston-rockets-vector-logo.png"
        },
        "Indiana Pacers": {
            "emoji": "🏁",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/indiana-pacers-vector-logo.png"
        },
        "LA Clippers": {
            "emoji": "⛵",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/la-clippers-vector-logo.png"
        },
        "Los Angeles Lakers": {
            "emoji": "🏀",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/los-angeles-lakers-vector-logo.png"
        },
        "Memphis Grizzlies": {
            "emoji": "🐻",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/memphis-grizzlies-vector-logo.png"
        },
        "Miami Heat": {
            "emoji": "🔥",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/miami-heat-vector-logo.png"
        },
        "Milwaukee Bucks": {
            "emoji": "🦌",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/milwaukee-bucks-vector-logo.png"
        },
        "Minnesota Timberwolves": {
            "emoji": "🐺",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/minnesota-timberwolves-vector-logo.png"
        },
        "New Orleans Pelicans": {
            "emoji": "🦢",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/new-orleans-pelicans-vector-logo.png"
        },
        "New York Knicks": {
            "emoji": "🗽",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/new-york-knicks-vector-logo.png"
        },
        "Oklahoma City Thunder": {
            "emoji": "⚡",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/oklahoma-city-thunder-vector-logo.png"
        },
        "Orlando Magic": {
            "emoji": "🎭",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/orlando-magic-vector-logo.png"
        },
        "Philadelphia 76ers": {
            "emoji": "🔔",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/philadelphia-76ers-vector-logo.png"
        },
        "Phoenix Suns": {
            "emoji": "☀️",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/phoenix-suns-vector-logo.png"
        },
        "Portland Trail Blazers": {
            "emoji": "🌲",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/portland-trail-blazers-vector-logo.png"
        },
        "Sacramento Kings": {
            "emoji": "👑",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/sacramento-kings-vector-logo.png"
        },
        "San Antonio Spurs": {
            "emoji": "🤠",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/san-antonio-spurs-vector-logo.png"
        },
        "Toronto Raptors": {
            "emoji": "🦖",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/toronto-raptors-vector-logo.png"
        },
        "Utah Jazz": {
            "emoji": "🎵",
            "conference": "West",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/utah-jazz-vector-logo.png"
        },
        "Washington Wizards": {
            "emoji": "🧙",
            "conference": "East",
            "logo": "https://logoeps.com/wp-content/uploads/2013/03/washington-wizards-vector-logo.png"
        }
    }

def format_currency(amount: float, currency: str = "€") -> str:
    """
    Formatta valuta con colori
    """
    color = "#27ae60" if amount >= 0 else "#e74c3c"
    sign = "+" if amount > 0 else ""
    return f'<span style="color: {color}; font-weight: bold;">{sign}{amount:.2f}{currency}</span>'

def create_responsive_metric(label: str, value: str, delta: Optional[str] = None, 
                           help_text: Optional[str] = None) -> None:
    """
    Metrica responsive per mobile
    """
    delta_html = f'<br><small style="color: #888;">{delta}</small>' if delta else ''
    help_html = f'<br><small style="color: #666; font-style: italic;">{help_text}</small>' if help_text else ''
    
    st.markdown(f"""
    <div style="
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    ">
        <h3 style="margin: 0; color: #1f4e79; font-size: 1.1em;">{label}</h3>
        <h2 style="margin: 0.5rem 0; color: #262730; font-size: 1.8em;">{value}</h2>
        {delta_html}
        {help_html}
    </div>
    """, unsafe_allow_html=True)