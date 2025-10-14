import streamlit as st
import pandas as pd
from rebalancing.policy import RebalancingPolicy

class StreamlitDashboard:
    def __init__(self):
        self.policy = RebalancingPolicy()

    def run(self, df_path="predictions.csv"):
        st.title("🚲 Vélib’Now — Disponibilité & Rééquilibrage Smart")
        df = pd.read_csv(df_path)
        st.map(df, latitude="coordonnees_geo.lat", longitude="coordonnees_geo.lon")
        st.write("### 🔧 Stations à rééquilibrer")
        st.dataframe(self.policy.suggest(df))
