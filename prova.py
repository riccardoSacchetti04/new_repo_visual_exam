import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Game Analytics – Executive Dashboard",
    page_icon="🎮",
    layout="wide"
)

st.sidebar.title("🎮 Game Analytics")
st.sidebar.caption("Prototipo interno – Team Game Analytics")

@st.cache_data
def load_data(path: str = r"C:\Users\RiccardoSacchetti\OneDrive - ITS Angelo Rizzoli\Documenti\GitHub\new_repo_visual_exam\vgsales_imputed.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df = df[df["Year_of_Release"] >= 2000]
    df["HIT"] = (df["Global_Sales"] >= 1.0).astype(int)
    return df

@st.cache_resource
def load_trained_model(path=r"C:\Users\RiccardoSacchetti\OneDrive - ITS Angelo Rizzoli\Documenti\GitHub\new_repo_visual_exam\modello_vgsales_ottimizzato1.pkl"):
    model = joblib.load(path)

    # Feature usate nel training
    feature_cols = [
        "Platform",
        "Genre",
        "Publisher",
        "Year_of_Release",
        "Critic_Score",
        "User_Score",
        "Rating",
    ]
    return model, feature_cols

def format_millions(x):
    return f"{x:,.1f}".replace(",", " ")

df = load_data()
min_year = int(df["Year_of_Release"].min())
max_year = int(df["Year_of_Release"].max())

model, feature_cols = load_trained_model()

# -----------------------------------------------------------
# NAVIGAZIONE
# -----------------------------------------------------------
page = st.sidebar.radio(
    "Sezioni",
    [
        "Overview",
        "Analisi Mercato",
        "Modello HIT Predictor",
        "Esplora dati",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: vendite e recensioni videogiochi dal 2000 in poi.")

# -----------------------------------------------------------
# 1. OVERVIEW BUSINESS
# -----------------------------------------------------------
if page == "Overview":
    st.title("Overview")
    st.markdown("### Performance storiche del catalogo")
    
    max_value_dataset = float(df["Global_Sales"].max())

    with st.container(border=True):
        #filtri
        col_year, col_min, col_max = st.columns([2, 1, 1])
        
        with col_year:
            year_range = st.slider(
                "Periodo di Analisi",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        
        with col_min:
            min_sales_filter = st.number_input(
                "Minimo di vendite", 
                min_value=0.0, 
                max_value=max_value_dataset,
                value=0.0, 
                step=0.5,
            
            )

        with col_max:
            max_sales_filter = st.number_input(
                "Massimo di vendite", 
                min_value=0.0, 
                max_value=max_value_dataset,
                value=max_value_dataset, 
                step=1.0,
        
            )


    mask = (
        (df["Year_of_Release"] >= year_range[0]) & 
        (df["Year_of_Release"] <= year_range[1]) &
        (df["Global_Sales"] >= min_sales_filter) &
        (df["Global_Sales"] <= max_sales_filter) 
    )
    df_filt = df[mask].copy()

    if df_filt.empty:
        st.warning("Nessun dato trovato con questi filtri.")
    else:
    
        st.markdown("#### KPI Principali")
        k1, k2, k3, k4 = st.columns(4)
        
        tot_sales = df_filt["Global_Sales"].sum()
        avg_critic = df_filt["Critic_Score"].mean()
        hit_count = df_filt[df_filt["Global_Sales"] >= 1.0].shape[0]
        hit_rate = (hit_count / len(df_filt)) * 100

        # Visualizzazione
        k1.metric("Totale copie", f"{tot_sales:,.1f} M")
        k2.metric("Volumi Rilasciati", f"{len(df_filt):,}")
        k3.metric("Quality Score (Media)", f"{avg_critic:.1f}/100")
        k4.metric("Tasso di HIT (>1M)", f"{hit_rate:.1f}%", help="Percentuale di giochi sopra il milione di copie")

        st.markdown("---")

        # grafic
        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            st.subheader("Trend Vendite Temporale")
            
            trend_data = df_filt.groupby("Year_of_Release")["Global_Sales"].sum()
            st.area_chart(trend_data, color="#FF4B4B")
            st.caption("Evoluzione del volume di vendita totale (Milioni di copie) anno su anno.")

        with col_chart2:
            st.subheader("Top Piattaforme")
        
            plat_sales = df_filt.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False)
            
            if len(plat_sales) > 5:
                top_5 = plat_sales.iloc[:5]
                others = pd.Series([plat_sales.iloc[5:].sum()], index=["Others"])
                plat_sales_clean = pd.concat([top_5, others])
            else:
                plat_sales_clean = plat_sales

            st.bar_chart(plat_sales_clean, color="#1F77B4", horizontal=True)
            st.caption("Quote di mercato per piattaforma (Top 5).")


# # -----------------------------------------------------------
# # PAGINA 2 – PIATTAFORME & GENERI
# # -----------------------------------------------------------
# elif page == "Piattaforme & Generi":

#     st.title("Piattaforme & Generi")

#     colf1, colf2 = st.columns(2)
#     with colf1:
#         year_range_pg = st.slider(
#             "Periodo",
#             min_value=min_year,
#             max_value=max_year,
#             value=(max_year - 5, max_year),
#             key="pg"
#         )
#     with colf2:
#         region = st.selectbox(
#             "Area vendite",
#             ["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
#         )

#     mask = (df["Year_of_Release"] >= year_range_pg[0]) & (df["Year_of_Release"] <= year_range_pg[1])
#     df_pg = df[mask].copy()

#     st.subheader("Top piattaforme per vendite")
#     stats = df_pg.groupby("Platform").agg(
#         sales=(region, "sum"),
#         n_games=("Name", "count"),
#         hit_rate=("HIT", "mean"),
#     ).reset_index()

#     top = stats.sort_values("sales", ascending=False).head(10)
#     st.bar_chart(top.set_index("Platform")["sales"])





# -----------------------------------------------------------
# PAGINA 4 – MODELLO HIT
# -----------------------------------------------------------
elif page == "Modello HIT Predictor":

    st.title("HIT Predictor – Modello ML")

    st.markdown("Compila la scheda prodotto per stimare la probabilità di HIT.")

    with st.form("hit_form"):
        c1, c2 = st.columns(2)

        with c1:
            platform = st.selectbox("Piattaforma", sorted(df["Platform"].unique()))
            genre = st.selectbox("Genere", sorted(df["Genre"].unique()))
            publisher = st.selectbox("Publisher", sorted(df["Publisher"].unique()))
            #rating = st.selectbox("Rating ESRB", sorted(df["Rating"].unique()))

        with c2:
            year = st.slider("Anno d'uscita previsto", min_year, max_year+10, max_year)
            critic_score = st.slider("Critic Score atteso", 0, 100, 80)
            user_score = st.slider("User Score atteso", 0.0, 10.0, 8.0, 0.1)

        submitted = st.form_submit_button("Stima probabilità HIT")

    if submitted:
            # 1. PREDIZIONE
            x_input = pd.DataFrame([{
                "Platform": platform,
                "Genre": genre,
                "Publisher": publisher,
                "Rating": df["Rating"].mode()[0], 
                "Year_of_Release": year,
                "Critic_Score": critic_score,
                "User_Score": df["User_Score"].median(), 
                "User_Count": df["User_Count"].median(),  
                "Critic_Count": df["Critic_Count"].median(), # 
                "Developer": "Unknown"
                
            }])
            
            try:
                pred_sales = model.predict(x_input)[0]
                if pred_sales < 0: pred_sales = 0.0
                
                
                st.markdown("### Risultato Stima")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.metric("Vendite Previste", f"{pred_sales:,.2f} M")
                
                with col_res2:
                    if pred_sales >= 1.0:
                        st.success("🎉 **PREVISIONE HIT!**")
                        st.progress(min(1.0, pred_sales / 5.0))
                    elif pred_sales >= 0.5:
                        st.warning("⚠️ **Performance Media.**")
                        st.progress(pred_sales / 3.0)
                    else:
                        st.error("📉 **Rischio Flop.**")
                        st.progress(max(0.05, pred_sales / 2.0))

                # confronto con esempi reali
                st.markdown("---")
                st.subheader(f"Lo scenario reale: {genre} su {platform}")
                st.caption("Esempi casuali di titoli simili che hanno avuto successo (HIT) o fallito (FLOP).")

                # filtro per genere e piattaforma
                mask_sim = (df["Genre"] == genre) & (df["Platform"] == platform)
                df_sim = df[mask_sim]
                
                
                if len(df_sim) < 10:
                    df_sim = df[df["Genre"] == genre]
                    st.caption(f"*(Dati piattaforma scarsi, mostro esempi globali del genere {genre})*")

              
                hits_pool = df_sim[df_sim["Global_Sales"] >= 1.0]
                flops_pool = df_sim[df_sim["Global_Sales"] < 0.5]

                
                hits_sample = hits_pool.sample(n=min(2, len(hits_pool))) if not hits_pool.empty else pd.DataFrame()
                flops_sample = flops_pool.sample(n=min(2, len(flops_pool))) if not flops_pool.empty else pd.DataFrame()

                # 4. Visualizzazione a Colonne
                col_h, col_f = st.columns(2)

                with col_h:
                    st.success("**Esempi di HIT** (Cosa vogliamo ottenere)")
                    if hits_sample.empty:
                        st.info("Nessun HIT registrato in questa categoria.")
                    else:
                        for _, row in hits_sample.iterrows():
                            with st.container(border=True):
                                st.markdown(f"**{row['Name']}** ({int(row['Year_of_Release'])})")
                                st.write(f"Vendite: **{row['Global_Sales']} M**")
                                st.write(f"Score: **{row['Critic_Score']:.0f}** | {row['Publisher']}")

                with col_f:
                    st.error("**Esempi di FLOP** (Cosa dobbiamo evitare)")
                    if flops_sample.empty:
                        st.info("Nessun FLOP registrato (o dati mancanti).")
                    else:
                        for _, row in flops_sample.iterrows():
                            with st.container(border=True):
                                st.markdown(f"**{row['Name']}** ({int(row['Year_of_Release'])})")
                                st.write(f"Vendite: **{row['Global_Sales']} M**")
                                st.write(f"Score: **{row['Critic_Score']:.0f}** | {row['Publisher']}")

            except Exception as e:
                st.error(f"Errore tecnico: {e}")

# -----------------------------------------------------------
# PAGINA 5 – Q&A
# -----------------------------------------------------------
elif page == "Esplora dati":

    st.title("Esplora dati")

    col1, col2, col3 = st.columns(3)
    with col1:
        pf = st.selectbox("Piattaforma", ["(Tutte)"] + sorted(df["Platform"].unique()))
    with col2:
        gn = st.selectbox("Genere", ["(Tutti)"] + sorted(df["Genre"].unique()))
    with col3:
        yr = st.selectbox("Anno", ["(Tutti)"] + sorted(df["Year_of_Release"].unique()))

    df_q = df.copy()
    if pf != "(Tutte)":
        df_q = df_q[df_q["Platform"] == pf]
    if gn != "(Tutti)":
        df_q = df_q[df_q["Genre"] == gn]
    if yr != "(Tutti)":
        df_q = df_q[df_q["Year_of_Release"] == yr]

    st.dataframe(df_q.sort_values("Global_Sales", ascending=False))

# -----------------------------------------------------------
# 6. ANALISI MERCATO (Strategia & Geografia)
# -----------------------------------------------------------
elif page == "Analisi Mercato":
    st.title("ANALISI DI MERCATO")
    st.markdown("Analisi Genere-Piattaforma")

    # --- CONTROLLI ---
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            # Filtro Regione (Cruciale per questa vista)
            region_col = st.selectbox(
                "Seleziona Mercato Focus",
                ["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales"],
                format_func=lambda x: {
                    "Global_Sales": "Mondo (Global)",
                    "NA_Sales": "Nord America",
                    "EU_Sales": "Europa",
                    "JP_Sales": "Giappone"
                }[x]
            )
        with c2:
            # Filtro Anno (per vedere l'evoluzione recente vs storica)
            year_market = st.slider("Periodo", min_year, max_year, (2010, max_year))

    # Filtro i dati
    df_m = df[(df["Year_of_Release"] >= year_market[0]) & (df["Year_of_Release"] <= year_market[1])].copy()

    # --- LAYOUT A COLONNE ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Top Generi")
        # Aggregazione per Genere
        genre_rank = df_m.groupby("Genre")[region_col].sum().sort_values(ascending=False).head(8)
        
        # Bar Chart orizzontale (più leggibile per le etichette lunghe)
        st.bar_chart(genre_rank, color="#2ECC71", horizontal=True)
    
    with col_right:
        st.subheader("Top Piattaforme")
        # Aggregazione per Piattaforma
        plat_rank = df_m.groupby("Platform")[region_col].sum().sort_values(ascending=False).head(8)
        st.bar_chart(plat_rank, color="#9B59B6", horizontal=True)

    st.markdown("---")

    st.subheader("HEATMAP: Generi vs Piattaforme")
    st.caption(f"La tabella mostra i volumi di vendita ({region_col.replace('_Sales', '')}) incrociati. I colori più scuri indicano le combinazioni più ricche.")


    top_genres = df_m.groupby("Genre")[region_col].sum().nlargest(10).index
    top_plats = df_m.groupby("Platform")[region_col].sum().nlargest(10).index
    
    df_pivot = df_m[df_m["Genre"].isin(top_genres) & df_m["Platform"].isin(top_plats)]
    
    matrix = df_pivot.pivot_table(
        index="Genre", 
        columns="Platform", 
        values=region_col, 
        aggfunc="sum",
        fill_value=0
    )

    
    st.dataframe(
        matrix.style.background_gradient(cmap="Reds", axis=None).format("{:.1f}"),
        use_container_width=True,
        height=400
    )
    st.title("Recensioni & successo commerciale")

    # year_range_r = st.slider(
    #     "Periodo",
    #     min_value=min_year,
    #     max_value=max_year,
    #     value=(max_year - 5, max_year),
    #     key="rev"
    # )

    df_r = df[(df["Year_of_Release"] >= year_market[0]) & (df["Year_of_Release"] <= year_market[1])]
    df_r = df_r.dropna(subset=["Critic_Score", "User_Score", "Global_Sales"])

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Critic Score vs Vendite")
        st.scatter_chart(df_r, x="Critic_Score", y="Global_Sales", color="#FF9F43")

    with colB:
        st.subheader("User Score vs Vendite")
        st.scatter_chart(df_r, x="User_Score", y="Global_Sales", color="#9B59B6")
    
    st.markdown("---")
    st.markdown("#### Vendite medie per fascia di valutazione critica")

    # Binning critic score
    df_r["Critic_Band"] = pd.cut(
        df_r["Critic_Score"],
        bins=[0, 60, 75, 85, 100],
        labels=["<60", "60–75", "75–85", ">85"],
        include_lowest=True,
    )

    band_stats = (
        df_r.groupby("Critic_Band")
        .agg(
            n_games=("Name", "count"),
            hit_rate=("HIT", "mean"),
        )
        .reset_index()
    )

    # st.bar_chart(
    #     data=band_stats.set_index("Critic_Band"),
    #     y="hit_rate",
    # )

    st.caption(
        "Mostra la **quota HIT (%)** per ciascuna fascia di Critic Score. "
        "Indica quanto spesso un gioco con quella fascia di valutazione supera 1M copie."
    )

    st.dataframe(
        band_stats.rename(
            columns={
                "Critic_Band": "Fascia Critic Score",
                "n_games": "N. giochi",
                "hit_rate": "Quota HIT",
            }
        ).style.format(
            {
                "Quota HIT": "{:0.2%}",
            }
        ),
        use_container_width=True,
    )



