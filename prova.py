import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


# -----------------------------------------------------------
# CONFIGURAZIONE BASE
# -----------------------------------------------------------
st.set_page_config(
    page_title="Game Analytics – Executive Dashboard",
    page_icon="🎮",
    layout="wide"
)

st.sidebar.title("🎮 Game Analytics")
st.sidebar.caption("Prototipo interno – Team Game Analytics")


# -----------------------------------------------------------
# FUNZIONI DI SUPPORTO
# -----------------------------------------------------------
@st.cache_data
def load_data(path: str = r"C:\Users\MarcoLeonida\OneDrive - ITS Angelo Rizzoli\Documenti\GitHub\new_repo_visual_exam\vgsales_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalizza nomi colonne se servisse
    df.columns = [c.strip() for c in df.columns]

    # Rimuovi anni pre-2000 se presenti per sicurezza
    df = df[df["Year_of_Release"] >= 2000]

    # Crea variabile target HIT (Global_Sales in milioni)
    df["HIT"] = (df["Global_Sales"] >= 1.0).astype(int)
    return df


@st.cache_resource
def train_hit_model(df: pd.DataFrame):
    """
    Allena un modello semplice per classificare HIT vs non-HIT.
    Ritorna pipeline sklearn già pronta per la predizione + metriche.
    """
    # Selezione feature (solo quelle realistiche disponibili prima del lancio)
    feature_cols = [
        "Platform",
        "Genre",
        "Publisher",
        "Year_of_Release",
        "Critic_Score",
        "User_Score",
        "Rating",
    ]

    data = df.dropna(subset=feature_cols + ["HIT"]).copy()

    X = data[feature_cols]
    y = data["HIT"]

    cat_cols = ["Platform", "Genre", "Publisher", "Rating"]
    num_cols = ["Year_of_Release", "Critic_Score", "User_Score"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    hit_rate = y_test.mean()

    metrics = {
        "AUC": auc,
        "Accuracy": acc,
        "Test_HIT_rate": hit_rate,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return pipe, feature_cols, metrics


def format_millions(x):
    return f"{x:,.1f}".replace(",", " ")


# -----------------------------------------------------------
# CARICAMENTO DATI
# -----------------------------------------------------------
df = load_data()

# Precalcoli base
min_year = int(df["Year_of_Release"].min())
max_year = int(df["Year_of_Release"].max())


# -----------------------------------------------------------
# SIDEBAR – NAVIGAZIONE
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


# -----------------------------------------------------------
# PAGINA 1 – OVERVIEW & KPI
# -----------------------------------------------------------
if page == "Overview":
    st.title("Executive Overview")

    st.markdown(
        """
        Questa pagina fornisce una vista di alto livello sul catalogo:
        **dimensione**, **andamento delle vendite** e **qualità media** nel tempo.
        """
    )

    # Filtro anno
    with st.sidebar:
        st.subheader("Filtri comuni")
        year_range = st.slider(
            "Periodo di analisi (anno di uscita)",
            min_value=min_year,
            max_value=max_year,
            value=(max_year - 5, max_year),
            step=1,
        )

    mask_year = (df["Year_of_Release"] >= year_range[0]) & (
        df["Year_of_Release"] <= year_range[1]
    )
    df_filt = df[mask_year].copy()

    # KPI in alto
    col1, col2, col3, col4 = st.columns(4)

    total_games = len(df_filt)
    total_sales = df_filt["Global_Sales"].sum()
    avg_critic = df_filt["Critic_Score"].mean()
    hit_rate = df_filt["HIT"].mean() * 100

    col1.metric("Giochi nel catalogo (periodo)", f"{total_games}")
    col2.metric("Vendite globali (M copie)", format_millions(total_sales))
    col3.metric("Critic Score medio", f"{avg_critic:0.1f}" if not np.isnan(avg_critic) else "n.d.")
    col4.metric("Quota HIT (≥1M copie)", f"{hit_rate:0.1f}%")

    st.markdown("---")

    # Andamento vendite per anno
    st.subheader("Andamento vendite globali nel tempo")

    sales_by_year = (
        df.groupby("Year_of_Release")["Global_Sales"]
        .sum()
        .reset_index()
        .sort_values("Year_of_Release")
    )

    st.line_chart(
        data=sales_by_year.set_index("Year_of_Release"),
        y="Global_Sales",
    )

    st.caption(
        f"**Come leggerlo:** mostra le vendite globali totali (milioni di copie) per anno di uscita, "
        f"dal {min_year} al {max_year}. I picchi indicano anni di particolare successo commerciale."
    )

    # Note sui dati
    with st.expander("Note sui dati"):
        st.markdown(
            """
            - Le vendite sono espresse in **milioni di copie**.
            - Il target **HIT** è definito come *Global_Sales ≥ 1M copie*.
            - Alcuni giochi potrebbero avere **valutazioni utenti/critica mancanti**, in particolare per titoli più vecchi o di nicchia.
            """
        )


# -----------------------------------------------------------
# PAGINA 2 – PIATTAFORME & GENERI
# -----------------------------------------------------------
elif page == "Piattaforme & Generi":
    st.title("Piattaforme & Generi")

    st.markdown(
        """
        Qui analizziamo **su quali piattaforme** e **in quali generi** il catalogo ha funzionato meglio,
        in termini di vendite globali e quota di HIT.
        """
    )

    # Filtri
    colf1, colf2 = st.columns(2)
    with colf1:
        year_range_pg = st.slider(
            "Periodo di analisi (anno di uscita)",
            min_value=min_year,
            max_value=max_year,
            value=(max_year - 5, max_year),
            step=1,
            key="pg_year_range",
        )
    with colf2:
        region = st.selectbox(
            "Area di vendite",
            ["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"],
            format_func=lambda x: {
                "Global_Sales": "Globale",
                "NA_Sales": "Nord America",
                "EU_Sales": "Europa",
                "JP_Sales": "Giappone",
                "Other_Sales": "Resto del mondo",
            }[x],
        )

    mask_year_pg = (df["Year_of_Release"] >= year_range_pg[0]) & (
        df["Year_of_Release"] <= year_range_pg[1]
    )
    df_pg = df[mask_year_pg].copy()

    # Top piattaforme
    st.subheader("Top piattaforme per vendite nel periodo selezionato")

    platform_stats = (
        df_pg.groupby("Platform")
        .agg(
            sales=(region, "sum"),
            n_games=("Name", "count"),
            hit_rate=("HIT", "mean"),
        )
        .reset_index()
    )

    # Mostra solo piattaforme con un minimo di giochi per evitare rumore
    min_games = st.slider(
        "Filtra piattaforme con almeno N giochi nel periodo",
        min_value=5,
        max_value=200,
        value=20,
        step=5,
    )
    platform_stats = platform_stats[platform_stats["n_games"] >= min_games]

    top_platforms = platform_stats.sort_values("sales", ascending=False).head(10)

    st.bar_chart(
        data=top_platforms.set_index("Platform"),
        y="sales",
    )

    st.caption(
        "Le barre mostrano le **vendite totali** (in milioni di copie) per piattaforma "
        f"nell'area **{region.replace('_Sales', '')}** nel periodo selezionato. "
        "Piattaforme con pochi titoli sono filtrate per evitare confronti fuorvianti."
    )

    # Tab per dettagli piattaforme / generi
    tab1, tab2 = st.tabs(["📦 Dettaglio piattaforme", "🎭 Analisi generi"])

    with tab1:
        st.markdown("#### Quota HIT per piattaforma")
        top_platforms["hit_rate_pct"] = top_platforms["hit_rate"] * 100
        st.dataframe(
            top_platforms[["Platform", "n_games", "sales", "hit_rate_pct"]]
            .rename(
                columns={
                    "Platform": "Piattaforma",
                    "n_games": "N. giochi",
                    "sales": f"Vendite ({region})",
                    "hit_rate_pct": "Quota HIT (%)",
                }
            )
            .style.format(
                {
                    f"Vendite ({region})": "{:,.1f}",
                    "Quota HIT (%)": "{:0.1f}",
                }
            ),
            use_container_width=True,
        )

    with tab2:
        st.markdown("#### Generi più performanti (vendite globali)")
        genre_stats = (
            df_pg.groupby("Genre")
            .agg(
                sales=("Global_Sales", "sum"),
                n_games=("Name", "count"),
                hit_rate=("HIT", "mean"),
            )
            .reset_index()
            .sort_values("sales", ascending=False)
        )

        st.bar_chart(
            data=genre_stats.set_index("Genre").head(10),
            y="sales",
        )

        st.caption(
            "Mostra i **10 generi** con maggiori vendite globali nel periodo selezionato. "
            "La scelta del genere impatta sia sul volume potenziale che sulla probabilità di ottenere un HIT."
        )


# -----------------------------------------------------------
# PAGINA 3 – RECENSIONI & VENDITE
# -----------------------------------------------------------
elif page == "Recensioni & Vendite":
    st.title("Recensioni & successo commerciale")

    st.markdown(
        """
        In questa sezione valutiamo **quanto contano** le recensioni di critica e utenti
        sul successo commerciale (vendite globali) dei giochi.
        """
    )

    # Filtro anno
    year_range_r = st.slider(
        "Periodo di analisi (anno di uscita)",
        min_value=min_year,
        max_value=max_year,
        value=(max_year - 5, max_year),
        step=1,
        key="rev_year_range",
    )

    mask_year_r = (df["Year_of_Release"] >= year_range_r[0]) & (
        df["Year_of_Release"] <= year_range_r[1]
    )
    df_r = df[mask_year_r].copy()

    # Rimuovi missing per analisi correlate
    df_r = df_r.dropna(subset=["Critic_Score", "User_Score", "Global_Sales"])

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Critic Score vs Vendite globali")
        st.scatter_chart(
            data=df_r,
            x="Critic_Score",
            y="Global_Sales",
        )
        corr_critic = df_r["Critic_Score"].corr(df_r["Global_Sales"])
        st.caption(
            f"Correlazione critica-vendite: **{corr_critic:0.2f}** (1 = relazione forte e positiva, 0 = nessuna relazione)."
        )

    with col_b:
        st.markdown("#### User Score vs Vendite globali")
        st.scatter_chart(
            data=df_r,
            x="User_Score",
            y="Global_Sales",
        )
        corr_user = df_r["User_Score"].corr(df_r["Global_Sales"])
        st.caption(
            f"Correlazione utenti-vendite: **{corr_user:0.2f}**. "
            "In generale, un valore positivo suggerisce che recensioni migliori sono associate a vendite maggiori."
        )

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
            avg_sales=("Global_Sales", "mean"),
            n_games=("Name", "count"),
            hit_rate=("HIT", "mean"),
        )
        .reset_index()
    )

    st.bar_chart(
        data=band_stats.set_index("Critic_Band"),
        y="avg_sales",
    )

    st.caption(
        "Mostra le **vendite medie per gioco** (milioni di copie) per fascia di Critic Score. "
        "Questo aiuta a quantificare l'impatto di recensioni eccellenti (>85) sul risultato commerciale."
    )

    st.dataframe(
        band_stats.rename(
            columns={
                "Critic_Band": "Fascia Critic Score",
                "avg_sales": "Vendite medie (M)",
                "n_games": "N. giochi",
                "hit_rate": "Quota HIT",
            }
        ).style.format(
            {
                "Vendite medie (M)": "{:0.2f}",
                "Quota HIT": "{:0.2%}",
            }
        ),
        use_container_width=True,
    )


# -----------------------------------------------------------
# PAGINA 4 – MODELLO HIT PREDICTOR
# -----------------------------------------------------------
elif page == "Modello HIT Predictor":
    st.title("HIT Predictor – Modello di Machine Learning")

    st.markdown(
        """
        Questa pagina espone un **prototipo di modello ML** che stima la probabilità
        che un nuovo gioco diventi un **HIT**.

        **Definizione di HIT (business):**  
        Un gioco è considerato HIT se le sue **vendite globali** sono **≥ 1 milione di copie**.
        """
    )

    with st.expander("ℹ️ Come usare questa pagina", expanded=True):
        st.markdown(
            """
            1. Compila la scheda prodotto (piattaforma, genere, anno, valutazioni attese…).  
            2. Clicca su **Stima probabilità HIT**.  
            3. Interpreta il risultato con il **semaforo**:
               - 🟢 **Alto potenziale**  
               - 🟡 **Intermedio / da valutare**  
               - 🔴 **Basso potenziale**  
            """
        )

    # Allena modello (cache_resource evita di ricalcolare ogni volta)
    with st.spinner("Alleno il modello sui dati storici..."):
        model, feature_cols, model_metrics = train_hit_model(df)

    st.markdown("##### Qualità modello (su test set)")
    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("AUC (ROC)", f"{model_metrics['AUC']:.2f}")
    colm2.metric("Accuracy", f"{model_metrics['Accuracy']:.2f}")
    colm3.metric("Quota HIT nel test", f"{model_metrics['Test_HIT_rate']*100:0.1f}%")
    colm4.metric("N giochi (test)", f"{model_metrics['n_test']}")

    st.markdown("---")

    st.subheader("Scheda prodotto – input modello")

    # Valori di default ragionevoli
    default_platform = df["Platform"].value_counts().idxmax()
    default_genre = df["Genre"].value_counts().idxmax()
    default_publisher = df["Publisher"].value_counts().idxmax()
    default_rating = df["Rating"].value_counts().idxmax()

    with st.form("hit_form"):
        c1, c2 = st.columns(2)

        with c1:
            platform = st.selectbox(
                "Piattaforma",
                sorted(df["Platform"].dropna().unique()),
                index=sorted(df["Platform"].dropna().unique()).index(default_platform),
            )
            genre = st.selectbox(
                "Genere",
                sorted(df["Genre"].dropna().unique()),
                index=sorted(df["Genre"].dropna().unique()).index(default_genre),
            )
            publisher = st.selectbox(
                "Publisher",
                sorted(df["Publisher"].dropna().unique()),
                index=sorted(df["Publisher"].dropna().unique()).index(default_publisher),
            )
            rating = st.selectbox(
                "Rating ESRB",
                sorted(df["Rating"].dropna().unique()),
                index=sorted(df["Rating"].dropna().unique()).index(default_rating),
            )

        with c2:
            year = st.slider(
                "Anno di uscita previsto",
                min_value=min_year,
                max_value=max_year + 2,  # permetti leggera proiezione futuro
                value=max_year,
            )
            critic_score = st.slider(
                "Critic Score atteso (0-100)",
                min_value=0,
                max_value=100,
                value=80,
            )
            user_score = st.slider(
                "User Score atteso (0-10)",
                min_value=0.0,
                max_value=10.0,
                value=8.0,
                step=0.1,
            )

        submitted = st.form_submit_button("Stima probabilità HIT")

    if submitted:
        # Costruisci input per il modello
        x_new = pd.DataFrame(
            [
                {
                    "Platform": platform,
                    "Genre": genre,
                    "Publisher": publisher,
                    "Rating": rating,
                    "Year_of_Release": year,
                    "Critic_Score": critic_score,
                    "User_Score": user_score,
                }
            ]
        )

        proba_hit = model.predict_proba(x_new)[0, 1]
        proba_pct = proba_hit * 100

        # Semaforo
        if proba_hit >= 0.7:
            level = "🟢 Alto potenziale di HIT"
        elif proba_hit >= 0.4:
            level = "🟡 Potenziale intermedio (da valutare)"
        else:
            level = "🔴 Basso potenziale di HIT"

        st.markdown("### Risultato")
        st.metric("Probabilità di HIT stimata", f"{proba_pct:0.1f}%")
        st.markdown(f"**Interpretazione:** {level}")

        st.caption(
            "Questo è un prototipo interno: il modello si basa solo sullo storico disponibile "
            "e non tiene conto di budget marketing, concorrenza, IP possedute, ecc."
        )

    st.markdown("---")
    st.subheader("Esempi reali dal catalogo")

    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        st.markdown("**Esempio di HIT reale**")
        hit_sample = df[df["HIT"] == 1].sample(1, random_state=42)
        st.write(hit_sample[["Name", "Platform", "Genre", "Year_of_Release", "Global_Sales"]])
        x_hit = hit_sample[feature_cols]
        proba_hit_real = model.predict_proba(x_hit)[0, 1]
        st.caption(f"Probabilità stimata dal modello: **{proba_hit_real*100:0.1f}%**")

    with col_ex2:
        st.markdown("**Esempio di NON-HIT reale**")
        nonhit_sample = df[df["HIT"] == 0].sample(1, random_state=123)
        st.write(
            nonhit_sample[
                ["Name", "Platform", "Genre", "Year_of_Release", "Global_Sales"]
            ]
        )
        x_non = nonhit_sample[feature_cols]
        proba_non_real = model.predict_proba(x_non)[0, 1]
        st.caption(f"Probabilità stimata dal modello: **{proba_non_real*100:0.1f}%**")


# -----------------------------------------------------------
# PAGINA 5 – ESPLORA DATI / Q&A (BETA)
# -----------------------------------------------------------
elif page == "🔍 Esplora dati / Q&A (beta)":
    st.title("🔍 Esplora dati / Q&A (beta)")

    st.markdown(
        """
        Questa pagina è pensata come **entry point** per una futura integrazione LLM/GenAI,
        ma fornisce già una modalità semplice per interrogare il dataset.
        """
    )

    st.markdown("#### Filtro rapido")

    colq1, colq2, colq3 = st.columns(3)
    with colq1:
        platform_q = st.selectbox(
            "Piattaforma (opzionale)",
            ["(Tutte)"] + sorted(df["Platform"].dropna().unique().tolist()),
        )
    with colq2:
        genre_q = st.selectbox(
            "Genere (opzionale)",
            ["(Tutti)"] + sorted(df["Genre"].dropna().unique().tolist()),
        )
    with colq3:
        year_q = st.selectbox(
            "Anno (opzionale)",
            ["(Tutti)"] + sorted(df["Year_of_Release"].dropna().unique().astype(int).tolist()),
        )

    df_q = df.copy()
    if platform_q != "(Tutte)":
        df_q = df_q[df_q["Platform"] == platform_q]
    if genre_q != "(Tutti)":
        df_q = df_q[df_q["Genre"] == genre_q]
    if year_q != "(Tutti)":
        df_q = df_q[df_q["Year_of_Release"] == int(year_q)]

    st.markdown("#### Giochi filtrati")
    st.dataframe(
        df_q[
            [
                "Name",
                "Platform",
                "Genre",
                "Year_of_Release",
                "Global_Sales",
                "Critic_Score",
                "User_Score",
            ]
        ].sort_values("Global_Sales", ascending=False),
        use_container_width=True,
        height=400,
    )

    st.caption(
        "Suggerimento: puoi usare questa vista live durante la demo per rispondere a domande ad-hoc del management "
        "del tipo *“quali sono stati i top giochi Action su PS3 nel 2012?”*."
    )

    st.markdown("---")
    st.subheader("🔮 Sezione GenAI (placeholder)")

    st.markdown(
        """
        Qui si potrebbe integrare un **modello LLM** (es. tramite API o Ollama) per
        permettere domande in linguaggio naturale sul catalogo.  
        Esempi:
        - *“Mostrami i 5 giochi con le migliori vendite in Europa dal 2015 in poi.”*  
        - *“Quali generi hanno la quota di HIT più alta su Nintendo?”*  
        """
    )

    st.info(
        "Per l'esame puoi spiegare come collegheresti un LLM che traduce la domanda naturale "
        "in una query sui dati (es. usando un agente che genera filtri pandas)."
    )
