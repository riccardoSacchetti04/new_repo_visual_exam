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
def load_data(path: str = r"C:\Users\MarcoLeonida\OneDrive - ITS Angelo Rizzoli\Documenti\GitHub\new_repo_visual_exam\vgsales_imputed.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df = df[df["Year_of_Release"] >= 2000]
    df["HIT"] = (df["Global_Sales"] >= 1.0).astype(int)
    return df

@st.cache_resource
def load_trained_model(path=r"C:\Users\MarcoLeonida\OneDrive - ITS Angelo Rizzoli\Documenti\GitHub\new_repo_visual_exam\modello_vgsales_ottimizzato.pkl"):
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
        "Piattaforme & Generi",
        "Recensioni -> Vendite",
        "Modello HIT Predictor",
        "Esplora dati / Q&A (beta)",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: vendite e recensioni videogiochi dal 2000 in poi.")

# PAGINA 1 – OVERVIEW

if page == "Overview":
    st.title("Executive Overview")

    year_range = st.slider(
        "Periodo di analisi",
        min_value=min_year,
        max_value=max_year,
        value=(max_year - 5, max_year)
    )

    mask = (df["Year_of_Release"] >= year_range[0]) & (df["Year_of_Release"] <= year_range[1])
    df_filt = df[mask].copy()

    # KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giochi", f"{len(df_filt)}")
    col2.metric("Vendite globali (M)", format_millions(df_filt["Global_Sales"].sum()))
    col3.metric("Critic Score medio", f"{df_filt['Critic_Score'].mean():0.1f}")
    col4.metric("Quota HIT (%)", f"{df_filt['HIT'].mean()*100:0.1f}%")

    st.markdown("---")

    st.subheader("Vendite globali nel tempo")

    
    sales_by_year = (
        df.groupby("Year_of_Release")["Global_Sales"]
        .sum()
        .reset_index()
        .sort_values("Year_of_Release")
    )

    sales_by_year["Year_str"] = sales_by_year["Year_of_Release"].astype(int).astype(str)

    st.line_chart(
        data=sales_by_year.set_index("Year_str"),
        y="Global_Sales",
    )
    st.caption(
        f"**Come leggerlo:** mostra le vendite globali totali (milioni di copie) per anno di uscita, "
        f"dal {min_year} al {max_year}. I picchi indicano anni di particolare successo commerciale."
    )

# -----------------------------------------------------------
# PAGINA 2 – PIATTAFORME & GENERI
# -----------------------------------------------------------
elif page == "Piattaforme & Generi":

    st.title("Piattaforme & Generi")

    colf1, colf2 = st.columns(2)
    with colf1:
        year_range_pg = st.slider(
            "Periodo",
            min_value=min_year,
            max_value=max_year,
            value=(max_year - 5, max_year),
            key="pg"
        )
    with colf2:
        region = st.selectbox(
            "Area vendite",
            ["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
        )

    mask = (df["Year_of_Release"] >= year_range_pg[0]) & (df["Year_of_Release"] <= year_range_pg[1])
    df_pg = df[mask].copy()

    st.subheader("Top piattaforme per vendite")
    stats = df_pg.groupby("Platform").agg(
        sales=(region, "sum"),
        n_games=("Name", "count"),
        hit_rate=("HIT", "mean"),
    ).reset_index()

    top = stats.sort_values("sales", ascending=False).head(10)
    st.bar_chart(top.set_index("Platform")["sales"])


# -----------------------------------------------------------
# PAGINA 3 – RECENSIONI -> VENDITE
# -----------------------------------------------------------
elif page == "Recensioni -> Vendite":
    st.title("Recensioni & successo commerciale")

    year_range_r = st.slider(
        "Periodo",
        min_value=min_year,
        max_value=max_year,
        value=(max_year - 5, max_year),
        key="rev"
    )

    df_r = df[(df["Year_of_Release"] >= year_range_r[0]) & (df["Year_of_Release"] <= year_range_r[1])]
    df_r = df_r.dropna(subset=["Critic_Score", "User_Score", "Global_Sales"])

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Critic Score vs Vendite")
        st.scatter_chart(df_r, x="Critic_Score", y="Global_Sales")

    with colB:
        st.subheader("User Score vs Vendite")
        st.scatter_chart(df_r, x="User_Score", y="Global_Sales")


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
            rating = st.selectbox("Rating ESRB", sorted(df["Rating"].unique()))

        with c2:
            year = st.slider("Anno previsto", min_year, max_year+10, max_year)
            critic_score = st.slider("Critic Score atteso", 0, 100, 80)
            user_score = st.slider("User Score atteso", 0.0, 10.0, 8.0, 0.1)

        submitted = st.form_submit_button("Stima probabilità HIT")

    if submitted:
        x_new = pd.DataFrame([{
            "Platform": platform,
            "Genre": genre,
            "Publisher": publisher,
            "Rating": rating,
            "Year_of_Release": year,
            "Critic_Score": critic_score,
            "User_Score": user_score,
        }])

        x_new["User_Count"] = df["User_Count"].median()
        x_new["Developer"] = "Unknown"
        x_new["Critic_Count"] = df["Critic_Count"].median()

        x_new["NA_Sales"] = 0.0
        x_new["EU_Sales"] = 0.0
        x_new["JP_Sales"] = 0.0
        x_new["Other_Sales"] = 0.0


        pred = model.predict(x_new)[0]
        pct = pred * 10000
        st.caption(pred)

        st.metric("Probabilità HIT stimata", f"{pct:.1f}%")

        if pred >= 70:
            st.success("🟢 Alto potenziale di HIT!")
        elif pred >= 40:
            st.warning("🟡 Potenziale intermedio.")
        else:
            st.error("🔴 Basso potenziale di HIT.")

        st.markdown("---")
        st.subheader("Esempi reali con probabilità superiore")

        df_ex = df.dropna(subset=feature_cols).copy()

        # AGGIUNGI LE COLONNE MANCANTI PERCHÉ IL MODELLO LE RICHIEDE
        df_ex["User_Count"] = df["User_Count"].median()
        df_ex["Developer"] = "Unknown"
        df_ex["Critic_Count"] = df["Critic_Count"].median()

        df_ex["NA_Sales"] = 0.0
        df_ex["EU_Sales"] = 0.0
        df_ex["JP_Sales"] = 0.0
        df_ex["Other_Sales"] = 0.0

        # Ora il modello ha TUTTE le feature
        df_ex["proba"] = model.predict(df_ex).astype(float)


        better = df_ex[df_ex["proba"] > pred]

        if better.empty:
            st.info("Nessun gioco supera la tua configurazione.")
        else:
            top = better.nlargest(2, "proba")
            col1, col2 = st.columns(2)

            for col, (_, row) in zip((col1, col2), top.iterrows()):
                with col:
                    st.markdown(f"**{row['Name']}**")
                    st.write(row[["Platform", "Genre", "Year_of_Release", "Global_Sales"]])
                    st.markdown(f"<span style='color:green;'>Probabilità: <b>{row['proba']*100:.1f}%</b></span>", unsafe_allow_html=True)


# -----------------------------------------------------------
# PAGINA 5 – Q&A
# -----------------------------------------------------------
elif page == "Esplora dati / Q&A (beta)":

    st.title("Esplora dati / Q&A (beta)")

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
