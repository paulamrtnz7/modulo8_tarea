# app.py

import streamlit as st
import pandas as pd
import requests
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ====================== LOGIN ======================
USUARIO_VALIDO = 'admin'
CONTRASENA_VALIDA = 'admin'

def login():
    st.title('Inicio de Sesión')
    usuario = st.text_input('Usuario', '')
    contrasena = st.text_input('Contraseña', '', type='password')

    if st.button('Iniciar sesión'):
        if usuario == USUARIO_VALIDO and contrasena == CONTRASENA_VALIDA:
            st.session_state.logged_in = True
            st.success('¡Inicio de sesión exitoso!')
        else:
            st.session_state.logged_in = False
            st.error('Usuario o contraseña incorrectos')

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ====================== CARGAR DATOS ======================
@st.cache_data
def cargar_datos():
    return pd.read_csv("data/FUTFEM_Datos.csv", sep=';')

df = cargar_datos()

# ====================== MENÚ ======================
st.sidebar.title("Menú")
menu = st.sidebar.selectbox("Selecciona una página", ("Inicio", "Estadísticas CSV", "Estadísticas API", "Modelo de Clasificación"))

# ====================== INICIO ======================
if menu == "Inicio":
    st.title("Bienvenido a la aplicación de fútbol")
    st.write("Explora estadísticas y datos de jugadoras y jugadores profesionales.")

# ====================== ESTADÍSTICAS CSV ======================
elif menu == "Estadísticas CSV":
    st.title("Estadísticas CSV")

    posiciones = df['POSICION'].dropna().unique()
    posicion_filtro = st.sidebar.selectbox("Posición", ["Todas"] + list(posiciones))

    equipos = df['EQUIPO'].dropna().unique()
    equipo_filtro = st.sidebar.selectbox("Equipo", ["Todos"] + list(equipos))

    edad_filtro = st.sidebar.slider("Edad", int(df['EDAD'].min()), int(df['EDAD'].max()), 
                                     (int(df['EDAD'].min()), int(df['EDAD'].max())))

    goles_filtro = st.sidebar.slider("Goles", int(df['GOLES'].min()), int(df['GOLES'].max()),
                                     (int(df['GOLES'].min()), int(df['GOLES'].max())))

    metricas_disponibles = [
        'GOLES', 'ASISTENCIAS', 'xG', 'xA', 'TIROS', 'TIROS A PUERTA',
        'PARTIDOS JUGADOS', 'TOQUES AREA PENAL', 'PASES COMPLETADOS',
        'REGATES COMPLETADOS', 'INTERCEPCIONES (Int)', 'FALTAS RECIBIDAS (Fld)',
        'DUELOS AEREOS GANADOS'
    ]
    selected_metrics = st.multiselect("Selecciona métricas", metricas_disponibles)

    df_filtrado = df.copy()
    if posicion_filtro != "Todas":
        df_filtrado = df_filtrado[df_filtrado['POSICION'] == posicion_filtro]
    if equipo_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado['EQUIPO'] == equipo_filtro]

    df_filtrado = df_filtrado[
        (df_filtrado['EDAD'] >= edad_filtro[0]) & (df_filtrado['EDAD'] <= edad_filtro[1]) &
        (df_filtrado['GOLES'] >= goles_filtro[0]) & (df_filtrado['GOLES'] <= goles_filtro[1])
    ]

    st.write(f"Jugadoras encontradas: {df_filtrado.shape[0]}")

    if selected_metrics:
        columnas_mostrar = ['JUGADOR', 'EDAD', 'POSICION', 'EQUIPO', 'NACIONALIDAD'] + selected_metrics
        st.dataframe(df_filtrado[columnas_mostrar])
    else:
        st.warning("Selecciona al menos una métrica para mostrar.")

    if not df_filtrado.empty:
        st.subheader("Visualizaciones")

        st.markdown("#### Top Goleadoras")
        top_goleadoras = df_filtrado.sort_values('GOLES', ascending=False).head(10)
        st.plotly_chart(px.bar(top_goleadoras, x='JUGADOR', y='GOLES', color='EQUIPO', text='GOLES'))

        st.markdown("#### Goles vs Partidos Jugados")
        st.plotly_chart(px.scatter(df_filtrado, x='GOLES', y='PARTIDOS JUGADOS',
                                   color='POSICION', hover_data=['JUGADOR', 'EQUIPO']))

        st.markdown("#### Boxplot de Goles por Posición")
        fig, ax = plt.subplots()
        sns.boxplot(data=df_filtrado, x='POSICION', y='GOLES', palette='pastel', ax=ax)
        st.pyplot(fig)

    if st.button("Exportar a PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Estadísticas de Jugadoras", ln=True, align='C')

        for _, row in df_filtrado[columnas_mostrar].iterrows():
            fila = " | ".join(str(x) for x in row)
            pdf.cell(200, 10, txt=fila[:200], ln=True)

        ruta_pdf = "estadisticas_jugadoras.pdf"
        pdf.output(ruta_pdf)

        with open(ruta_pdf, "rb") as f:
            st.download_button("Descargar PDF", f, file_name="estadisticas_jugadoras.pdf")

# ====================== ESTADÍSTICAS API ======================
elif menu == "Estadísticas API":
    st.title("Estadísticas API - WSL (Inglaterra)")
    league_id = "4849"
    api_key = "3"

    url_teams = f"https://www.thesportsdb.com/api/v1/json/{api_key}/lookup_all_teams.php?id={league_id}"
    teams = requests.get(url_teams).json().get("teams", [])

    if teams:
        team_names = [t["strTeam"] for t in teams]
        team_selected = st.selectbox("Selecciona un equipo", team_names)

        team_info = next((t for t in teams if t["strTeam"] == team_selected), None)
        if team_info:
            st.markdown(f"📍 Estadio: {team_info['strStadium']}")
            st.markdown(f"📅 Fundado: {team_info['intFormedYear']}")
            st.markdown(f"📖 Descripción: {team_info['strDescriptionEN'][:300]}...")

            players_url = f"https://www.thesportsdb.com/api/v1/json/{api_key}/lookup_all_players.php?id={team_info['idTeam']}"
            players_data = requests.get(players_url).json().get("player", [])

            if players_data:
                st.markdown("### Plantilla del equipo")
                for p in players_data:
                    st.markdown(f"**{p['strPlayer']}** - {p['strPosition']} ({p['strNationality']})")
                    st.markdown(f"🧬 Nacimiento: {p['dateBorn']}")
                    if p['strThumb']:
                        st.image(p['strThumb'], width=100)
                    st.markdown("---")
            else:
                st.warning("No se encontraron jugadoras.")
