import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import ml_pipeline as mp
from content import FEATURE_ENG, FEATURE_INFO, METRIC_INFO, MODEL_INFO

st.set_page_config(
    page_title="Online Shoppers: Purchase Prediction", page_icon="🛒", layout="wide"
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MONTH_ORDER = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
LABELS = ["No purchase", "Purchase"]
COLORS = {"No purchase": "#8a94a6", "Purchase": "#e8590c"}

PAGES = [
    "01 · Project Introduction",
    "02 · Dataset Exploration",
    "03 · Exploratory Data Analysis",
    "04 · Data Cleaning & Preprocessing",
    "05 · Correlation Analysis",
    "06 · Feature Engineering",
    "07 · Categorical Encoding",
    "08 · Feature Scaling",
    "09 · Model Selection & Training",
    "10 · Model Evaluation",
    "11 · Model Comparison",
    "12 · Conclusion & Recommendation",
    "Extra · Test saved models on a CSV",
]


# ============================================================ data and state
@st.cache_data(show_spinner=False)
def read_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def load_raw() -> pd.DataFrame:
    for p in (ROOT / mp.DATA_FILE, ROOT / "data" / mp.DATA_FILE):
        if p.exists():
            return read_csv_path(str(p))
    st.warning(f"`{mp.DATA_FILE}` was not found next to app.py. Upload it to continue.")
    up = st.file_uploader(
        "Full dataset (online_shoppers_intention.csv)", type="csv", key="full_upload"
    )
    if up is None:
        st.stop()
    return read_csv_bytes(up.getvalue())

@st.cache_data(show_spinner="Preparing data...")
def build_data(raw: pd.DataFrame) -> dict:
    df = mp.clean(raw)
    fe = mp.engineer(df)
    enc = mp.encode(fe)
    X_train, X_test, y_train, y_test = mp.split(enc)
    return dict(
        df=df,
        fe=fe,
        enc=enc,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

def results_store() -> dict:
    return st.session_state.setdefault("results", {})

def train_model(name: str, D: dict):
    t0 = time.time()
    model, res = mp.fit_and_evaluate(
        name, D["X_train"], D["y_train"], D["X_test"], D["y_test"]
    )
    results_store()[name] = dict(model=model, res=res, seconds=time.time() - t0)

def results_df() -> pd.DataFrame:
    store = results_store()
    rows = {n: store[n]["res"]["metrics"] for n in mp.MODEL_NAMES if n in store}
    return (
        pd.DataFrame(rows).T[mp.METRICS] if rows else pd.DataFrame(columns=mp.METRICS)
    )

def selected_model() -> str:
    return st.session_state.get("sel", mp.MODEL_NAMES[0])

def go_to(page: str):
    st.session_state["nav"] = page

# ================================================================== helpers
def four_part(what, why, impact, tells):
    with st.container(border=True):
        a, b = st.columns(2)
        a.markdown(f"**What are we doing?**  \n{what}")
        b.markdown(f"**Why are we doing it?**  \n{why}")
        c, d = st.columns(2)
        c.markdown(f"**What is its impact?**  \n{impact}")
        d.markdown(f"**What do the results tell us?**  \n{tells}")

def cm_fig(cm, title="", height=380):
    fig = px.imshow(
        cm,
        text_auto=True,
        x=LABELS,
        y=LABELS,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(
        title=title, coloraxis_showscale=False, height=height, margin=dict(t=50, b=10)
    )
    return fig

def roc_fig(curves: dict, aucs: dict, height=430):
    fig = go.Figure()
    for name, (fpr, tpr) in curves.items():
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={aucs[name]:.3f})"
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random guess",
            line=dict(dash="dash", color="gray"),
        )
    )
    fig.update_layout(
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=height,
        margin=dict(t=30, b=10),
    )
    return fig

def band(value, cuts, words):
    for cut, word in zip(cuts, words):
        if value >= cut:
            return word
    return words[-1]

def pct(x):
    return f"{x:.1%}"

def nav_buttons(i: int):
    st.divider()
    c1, c2 = st.columns(2)
    if i > 0:
        c1.button(
            f"← {PAGES[i-1]}", on_click=go_to, args=(PAGES[i - 1],), key=f"prev{i}"
        )
    if i < len(PAGES) - 1:
        c2.button(
            f"{PAGES[i+1]} →", on_click=go_to, args=(PAGES[i + 1],), key=f"next{i}"
        )

def metric_row(m: dict):
    cols = st.columns(6)
    for c, k in zip(cols, mp.METRICS):
        c.metric(k, f"{m[k]:.4f}")

def need_trained(name, D, key):
    """Ask the user to train `name` first. Returns True if it is trained."""
    if name in results_store():
        return True
    st.info(f"**{name}** has not been trained yet. Train it here or on page 09.")
    if st.button(f"Train {name}", type="primary", key=key):
        with st.spinner(f"Training {name}..."):
            train_model(name, D)
        st.rerun()
    return False

# ==================================================================== pages
def page_intro(raw, D):
    st.title("🛒 Predicting Online Shoppers' Purchase Intention")
    st.markdown(
        "An interactive walkthrough of a complete machine learning project: from raw website "
        "sessions to five trained classifiers that you can run, evaluate and compare."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("The problem")
        st.markdown(
            "Every row in the data is **one visit to an online shop**. We want to predict whether the "
            "visit ends in a **purchase** (`Revenue = 1`) or the visitor **leaves without buying** "
            "(`Revenue = 0`). This is a binary classification problem."
        )
        st.subheader("Why it matters")
        st.markdown(
            "- **Marketing spend:** show offers only to visitors who need a nudge, not to those who will buy anyway.\n"
            "- **Live help:** open a chat or discount for visitors who look likely to leave.\n"
            "- **Stock and planning:** understand which traffic and seasons produce sales.\n"
            "- **Site design:** find which browsing signals separate buyers from browsers."
        )
    with c2:
        st.subheader("How this app works")
        st.markdown(
            "Follow the steps in the sidebar in order. Each step explains what is being done, why, "
            "its impact, and what the results say. All numbers on the pages are computed live from the data."
        )
        st.markdown(
            "1. Explore and clean the data (steps 02-04)\n2. Understand features (steps 05-08)\n"
            "3. **Train a model of your choice** (step 09)\n4. Evaluate, compare and conclude (steps 10-12)"
        )
        st.info(
            "Only about 15% of visits end in a purchase, so accuracy alone can be misleading. "
            "We also use AUC, F1 and MCC."
        )
    st.subheader("The five models")
    cols = st.columns(5)
    for c, name in zip(cols, mp.MODEL_NAMES):
        with c.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(MODEL_INFO[name]["principle"])

def page_dataset(raw, D):
    st.header("Dataset Description and Exploration")
    st.markdown(
        "The **Online Shoppers Purchasing Intention** dataset (UCI / Kaggle) records one e-commerce "
        "session per row. Each session belongs to a different user over one year."
    )
    c = st.columns(4)
    c[0].metric("Records", f"{len(raw):,}")
    c[1].metric("Columns", raw.shape[1])
    c[2].metric("Input features", raw.shape[1] - 1)
    c[3].metric("Target", mp.TARGET)
    st.subheader("Target variable: `Revenue`")
    vc = raw[mp.TARGET].astype(int).value_counts()
    st.markdown(
        f"- `1` = purchase completed ({vc.get(1, 0):,} visits, {pct(vc.get(1, 0)/len(raw))})\n"
        f"- `0` = no purchase ({vc.get(0, 0):,} visits, {pct(vc.get(0, 0)/len(raw))})"
    )
    st.subheader("What each feature means")
    info = pd.DataFrame(
        FEATURE_INFO,
        columns=["Feature", "Type", "Meaning", "Why it helps predict purchase"],
    )
    st.dataframe(info, hide_index=True)
    st.subheader("Preview the original data")
    f1, f2, f3 = st.columns(3)
    outcome = f1.multiselect(
        "Outcome", ["No purchase", "Purchase"], default=["No purchase", "Purchase"]
    )
    months = f2.multiselect(
        "Month",
        sorted(
            raw["Month"].unique(),
            key=lambda m: MONTH_ORDER.index(m) if m in MONTH_ORDER else 99,
        ),
    )
    n = f3.slider("Rows to show", 5, 200, 20)
    view = raw[
        raw[mp.TARGET].astype(int).map({0: "No purchase", 1: "Purchase"}).isin(outcome)
    ]
    if months:
        view = view[view["Month"].isin(months)]
    st.caption(f"{len(view):,} rows match. Showing the first {min(n, len(view))}.")
    st.dataframe(view.head(n))
    st.subheader("Data types")
    st.dataframe(raw.dtypes.astype(str).rename("dtype").to_frame().T)

def page_eda(raw, D):
    st.header("Exploratory Data Analysis")
    st.markdown(
        "EDA means looking at the data before modelling. Structure, missing values and duplicates are "
        "checked on the original data. The charts further below use the de-duplicated data, as in the original notebook."
    )
    df = D["df"]
    t = st.tabs(
        [
            "Structure",
            "Missing values",
            "Duplicates",
            "Distributions",
            "Purchase vs no purchase",
            "Purchase rate by category",
        ]
    )

    with t[0]:
        info = pd.DataFrame(
            {
                "dtype": raw.dtypes.astype(str),
                "non-null": raw.notna().sum(),
                "unique values": raw.nunique(),
                "example": raw.iloc[0].astype(str),
            }
        )
        n_num = int(raw.select_dtypes("number").shape[1])
        four_part(
            "Listing each column with its data type, non-null count and number of unique values.",
            "Models need numbers, so we must know which columns are numeric, text or True/False.",
            "Tells us which columns need converting or encoding later.",
            f"{raw.shape[1]} columns: {n_num} numeric, {int((raw.dtypes == object).sum())} text "
            f"({', '.join(raw.columns[raw.dtypes == object])}) and {int((raw.dtypes == bool).sum())} True/False "
            f"({', '.join(raw.columns[raw.dtypes == bool])}). Text columns need encoding; True/False columns need converting to 1/0.",
        )
        st.dataframe(info)

    with t[1]:
        miss = raw.isna().sum()
        total = int(miss.sum())
        four_part(
            "Counting empty cells in every column.",
            "Most models cannot handle empty values, and many gaps can bias results.",
            "Decides whether we must fill or drop values before training.",
            (
                "No missing values, so no filling or dropping is needed."
                if total == 0
                else f"{total} missing cells found in {int((miss > 0).sum())} columns. They would need to be handled before training."
            ),
        )
        st.metric("Total missing cells", total)
        fig = px.bar(
            miss.reset_index(name="missing").rename(columns={"index": "column"}),
            x="column",
            y="missing",
        )
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig)

    with t[2]:
        d = int(raw.duplicated().sum())
        four_part(
            "Finding rows that are exact copies of an earlier row.",
            "Duplicates give repeated rows extra weight. If a copy lands in the test set while its twin is in the training set, "
            "the test score becomes too optimistic.",
            "We remove them in the cleaning step.",
            f"{d:,} duplicate rows ({pct(d/len(raw))} of the data).",
        )
        a, b = st.columns(2)
        a.metric("Duplicate rows", f"{d:,}")
        b.metric("Share of data", pct(d / len(raw)))
        if d:
            st.caption(
                "Sample of duplicated rows (all copies shown, grouped together):"
            )
            st.dataframe(
                raw[raw.duplicated(keep=False)].sort_values(list(raw.columns)).head(20)
            )

    with t[3]:
        sk = df[mp.NUM_COLS].skew().sort_values(ascending=False)
        four_part(
            "Plotting the distribution of each numeric column.",
            "The shape of a column shows outliers and skew. Skewed columns can hurt models that assume bell-shaped data or use distances.",
            "Motivates the log feature and the scaling step.",
            f"Most columns have many small values and a few very large ones. The most skewed is **{sk.index[0]}** "
            f"(skew {sk.iloc[0]:.1f}); the least is **{sk.index[-1]}** ({sk.iloc[-1]:.1f}).",
        )
        c1, c2, c3 = st.columns(3)
        col = c1.selectbox(
            "Column", mp.NUM_COLS, index=mp.NUM_COLS.index("ProductRelated_Duration")
        )
        bins = c2.slider("Bins", 10, 100, 40)
        split_by = c3.checkbox("Split by outcome", value=True)
        logy = c3.checkbox("Log count axis", value=False)
        d2 = df.assign(Outcome=df[mp.TARGET].map({0: LABELS[0], 1: LABELS[1]}))
        fig = px.histogram(
            d2,
            x=col,
            nbins=bins,
            color="Outcome" if split_by else None,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map=COLORS,
            log_y=logy,
        )
        fig.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig)
        st.dataframe(df[mp.NUM_COLS].describe().T.round(2))

    with t[4]:
        vc = df[mp.TARGET].value_counts().sort_index()
        buy = vc.get(1, 0) / vc.sum()
        four_part(
            "Counting purchasing and non-purchasing visits.",
            "If one class is much rarer, a model can look accurate by always predicting the common class.",
            "Tells us to rely on F1, MCC and AUC and not accuracy alone.",
            f"{pct(buy)} of visits buy. A model that always says 'no purchase' would be {pct(1-buy)} accurate "
            f"while finding no buyers. The data is imbalanced (about {vc.get(0,0)/max(vc.get(1,1),1):.1f} non-buyers per buyer).",
        )
        fig = px.bar(
            x=LABELS,
            y=[vc.get(0, 0), vc.get(1, 0)],
            color=LABELS,
            color_discrete_map=COLORS,
            text=[f"{vc.get(0,0):,}", f"{vc.get(1,0):,}"],
            labels=dict(x="", y="Visits"),
        )
        fig.update_layout(showlegend=False, height=360, margin=dict(t=10))
        st.plotly_chart(fig)

    with t[5]:
        c1, c2 = st.columns([1, 3])
        col = c1.selectbox("Category", mp.CAT_COLS)
        g = df.groupby(col)[mp.TARGET].agg(rate="mean", visits="count").reset_index()
        if col == "Month":
            g["o"] = g[col].map(
                lambda m: MONTH_ORDER.index(m) if m in MONTH_ORDER else 99
            )
            g = g.sort_values("o").drop(columns="o")
        else:
            g = g.sort_values(col)
        g[col] = g[col].astype(str)
        avg = df[mp.TARGET].mean()
        top, low = g.loc[g["rate"].idxmax()], g.loc[g["rate"].idxmin()]
        four_part(
            "Comparing the purchase rate across the groups of one categorical column.",
            "Shows which groups buy more than average, so we know the column carries information.",
            "Columns with clear differences are worth keeping and encoding.",
            f"Overall rate is {pct(avg)}. For **{col}**, the highest rate is **{top[col]}** ({pct(top['rate'])}, {int(top['visits']):,} visits) "
            f"and the lowest is **{low[col]}** ({pct(low['rate'])}, {int(low['visits']):,} visits)."
            "Small groups give noisy rates, so check the visit counts.",
        )
        fig = px.bar(
            g,
            x=col,
            y="rate",
            hover_data=["visits"],
            text=g["rate"].map(lambda v: f"{v:.0%}"),
        )
        fig.add_hline(
            y=avg, line_dash="dash", line_color="red", annotation_text="overall average"
        )
        fig.update_layout(height=400, yaxis_title="Purchase rate", margin=dict(t=10))
        c2.plotly_chart(fig)
        st.dataframe(g.assign(rate=g["rate"].round(3)), hide_index=True)

def page_cleaning(raw, D):
    st.header("Data Cleaning and Preprocessing")
    df = D["df"]
    st.subheader("Step 1: Remove duplicate rows")
    d = int(raw.duplicated().sum())
    four_part(
        "Dropping rows that are exact copies and resetting the row numbers.",
        "Copies over-weight some visits and can leak between training and test sets.",
        "A smaller but cleaner data set.",
        f"{d:,} rows removed, leaving {len(df):,}. The share of buyers moved from {pct(raw[mp.TARGET].astype(int).mean())} to {pct(df[mp.TARGET].mean())}.",
    )
    c = st.columns(4)
    c[0].metric("Rows before", f"{len(raw):,}")
    c[1].metric("Rows after", f"{len(df):,}", delta=f"{len(df)-len(raw):,}")
    c[2].metric("Duplicates before", f"{d:,}")
    c[3].metric("Duplicates after", int(df.duplicated().sum()))

    st.subheader("Step 2: Missing values")
    miss = int(raw.isna().sum().sum())
    if miss == 0:
        st.success(
            "No missing values were found, so nothing needs filling or dropping."
        )
    else:
        st.warning(
            f"{miss} missing cells found. The original notebook does not fill them, so rows with gaps would need handling before training."
        )
        st.dataframe(raw.isna().sum()[lambda s: s > 0].rename("missing").to_frame())
    st.caption("Why: models cannot learn from empty cells.")

    st.subheader("Step 3: Convert True/False columns into 1/0")
    four_part(
        "Changing `Revenue` and `Weekend` from True/False to 1/0.",
        "Models work on numbers. `Revenue` also becomes the numeric target.",
        "Same information, now in a form the models can use.",
        "Values are unchanged in meaning: True becomes 1 and False becomes 0.",
    )
    a, b = st.columns(2)
    with a:
        st.markdown("**Before**")
        st.dataframe(raw[["Weekend", mp.TARGET]].head(6))
        st.caption(
            f"dtypes: Weekend = {raw['Weekend'].dtype}, Revenue = {raw[mp.TARGET].dtype}"
        )
    with b:
        st.markdown("**After**")
        st.dataframe(df[["Weekend", mp.TARGET]].head(6))
        st.caption(
            f"dtypes: Weekend = {df['Weekend'].dtype}, Revenue = {df[mp.TARGET].dtype}"
        )

    st.subheader("Step 4: Prepare numeric and categorical features")
    a, b = st.columns(2)
    a.markdown(
        "**Numeric columns** (used as they are, and later scaled for most models):"
    )
    a.code(", ".join(mp.NUM_COLS))
    b.markdown("**Categorical columns** (explored in EDA):")
    b.code(", ".join(mp.CAT_COLS))
    st.markdown(
        "`Month` and `VisitorType` are text, so they are one-hot encoded in step 07. `OperatingSystems`, `Browser`, `Region` and "
        "`TrafficType` are category codes stored as numbers. As in the original notebook they stay as numbers."
    )
    st.markdown("**Summary of the cleaned numeric columns**")
    st.dataframe(df[mp.NUM_COLS].describe().T.round(2))

def page_corr(raw, D):
    st.header("Correlation Analysis")
    df = D["df"]
    corr = df[mp.NUM_COLS + [mp.TARGET]].corr()
    tgt = corr[mp.TARGET].drop(mp.TARGET).sort_values(ascending=False)
    pairs = (
        corr.drop(index=mp.TARGET, columns=mp.TARGET)
        .where(np.triu(np.ones((len(mp.NUM_COLS),) * 2), 1).astype(bool))
        .stack()
    )
    pair = pairs.abs().idxmax()
    four_part(
        "Measuring how strongly each pair of numeric columns moves together (Pearson correlation, -1 to +1).",
        "Shows which columns relate to buying, and which columns repeat each other.",
        "Strongly related columns are good predictors. Columns that mirror each other add little new information.",
        f"The strongest link to `Revenue` is **{tgt.abs().idxmax()}** ({corr.loc[tgt.abs().idxmax(), mp.TARGET]:+.2f}). "
        f"The most similar pair of inputs is **{pair[0]} and {pair[1]}** ({pairs[pair]:+.2f}).",
    )
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=620, margin=dict(t=10))
    st.plotly_chart(fig)
    st.subheader("Correlation with `Revenue`")
    bar = px.bar(
        tgt.reset_index(name="corr").rename(columns={"index": "feature"}),
        x="corr",
        y="feature",
        orientation="h",
        color="corr",
        color_continuous_scale="RdBu_r",
        range_color=(-1, 1),
    )
    bar.update_layout(
        height=400,
        margin=dict(t=10),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(bar)
    pos, neg = tgt[tgt > 0.05], tgt[tgt < -0.05]
    st.markdown("**How to read it**")
    st.markdown(
        f"- **Positive** ({', '.join(f'`{k}` {v:+.2f}' for k, v in pos.items()) or 'none above 0.05'}): when the value rises, buying is more likely.\n"
        f"- **Negative** ({', '.join(f'`{k}` {v:+.2f}' for k, v in neg.items()) or 'none below -0.05'}): when the value rises, buying is less likely. "
        "Visitors who leave quickly rarely buy.\n"
        "- Correlation measures **straight-line** links only, and it does not prove cause. A value near 0 can still hide a useful non-linear pattern."
    )
    st.info(
        "`PageValues` is computed from transaction outcomes, so its high correlation is partly because it already 'knows' the answer."
    )

def page_fe(raw, D):
    st.header("Feature Engineering")
    df, fe = D["df"], D["fe"]
    four_part(
        "Creating 7 new columns from the existing ones.",
        "Totals, averages, flags and a log transform give the models clearer clues than the raw columns do.",
        "More informative inputs, without collecting any new data.",
        "Open each feature below to see what it is, how it is calculated, and how it relates to buying.",
    )
    for feat, meta in FEATURE_ENG.items():
        corr_val = fe[feat].corr(fe[mp.TARGET])
        with st.expander(f"{feat}  (correlation with Revenue: {corr_val:+.3f})"):
            st.markdown(
                f"**What it is:** {meta['what']}  \n**How it is calculated:** `{meta['formula']}`  \n**Why:** {meta['why']}"
            )
            a, b = st.columns([1, 1])
            with a:
                st.markdown("**Original columns vs new feature**")
                st.dataframe(fe[meta["cols"] + [feat]].head(8).round(2))
                if feat == "log_ProductRelated_Duration":
                    st.caption(
                        f"Skew before: {fe['ProductRelated_Duration'].skew():.2f}. Skew after the log: {fe[feat].skew():.2f}."
                    )
            with b:
                if fe[feat].nunique() <= 2:
                    g = fe.groupby(feat)[mp.TARGET].mean().reset_index()
                    g[feat] = g[feat].astype(str)
                    fig = px.bar(
                        g,
                        x=feat,
                        y=mp.TARGET,
                        text=g[mp.TARGET].map(lambda v: f"{v:.1%}"),
                        labels={mp.TARGET: "Purchase rate"},
                    )
                else:
                    d2 = fe.assign(
                        Outcome=fe[mp.TARGET].map({0: LABELS[0], 1: LABELS[1]})
                    )
                    fig = px.histogram(
                        d2,
                        x=feat,
                        color="Outcome",
                        nbins=40,
                        histnorm="percent",
                        barmode="overlay",
                        opacity=0.7,
                        color_discrete_map=COLORS,
                    )
                fig.update_layout(height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig)
    st.subheader("Do the new features relate to the target?")
    nc = fe[mp.NEW_FEATS].corrwith(fe[mp.TARGET]).sort_values()
    orig = df[mp.NUM_COLS].corrwith(df[mp.TARGET])
    best_new = nc.abs().idxmax()
    fig = px.bar(
        nc.reset_index(name="corr").rename(columns={"index": "feature"}),
        x="corr",
        y="feature",
        orientation="h",
    )
    fig.update_layout(height=340, margin=dict(t=10))
    st.plotly_chart(fig)
    cmp = "stronger" if abs(nc[best_new]) > orig.abs().max() else "weaker"
    st.markdown(
        f"The strongest new feature is **{best_new}** ({nc[best_new]:+.3f}), which is {cmp} than the best original column "
        f"**{orig.abs().idxmax()}** ({orig[orig.abs().idxmax()]:+.3f}). "
        "`HasPageValue` is just `PageValues` turned into a yes/no flag, so it carries the same strong signal."
    )
    st.markdown("**Full result: original data plus new features**")
    st.dataframe(fe[mp.NEW_FEATS].head(10).round(2))

def page_encoding(raw, D):
    st.header("Categorical Feature Encoding")
    fe, enc = D["fe"], D["enc"]
    four_part(
        "One-hot encoding `Month` and `VisitorType`: each category becomes its own 0/1 column.",
        "Models only understand numbers. Coding months as 1 to 10 would wrongly say that some months are 'bigger' than others.",
        "Text becomes numeric columns with no fake ordering. The first category is dropped (`drop_first=True`) because it is implied when all the others are 0.",
        f"The data grows from {fe.shape[1]} to {enc.shape[1]} columns.",
    )
    c = st.columns(3)
    c[0].metric("Columns before", fe.shape[1])
    c[1].metric("Columns after", enc.shape[1], delta=enc.shape[1] - fe.shape[1])
    c[2].metric("Rows", f"{len(enc):,}")
    col = st.radio("Column to inspect", ["Month", "VisitorType"], horizontal=True)
    cats = sorted(fe[col].unique())
    dummies = [c for c in enc.columns if c.startswith(col + "_")]
    st.markdown(
        f"**{col}** has {len(cats)} categories: {', '.join(cats)}. The dropped baseline is **{cats[0]}** "
        f"(a visit from {cats[0]} has all {len(dummies)} columns equal to 0)."
    )
    mapping = (
        pd.concat([fe[col], enc[dummies]], axis=1)
        .drop_duplicates()
        .sort_values(col)
        .reset_index(drop=True)
    )
    st.markdown("**Category → encoded columns**")
    st.dataframe(mapping, hide_index=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Before** (text)")
        st.dataframe(fe[["Month", "VisitorType"]].head(8))
    with b:
        st.markdown("**After** (0/1 columns)")
        st.dataframe(
            enc[
                [c for c in enc.columns if c.startswith(("Month_", "VisitorType_"))]
            ].head(8)
        )
    with st.expander("Whole dataset before and after"):
        st.markdown("Before encoding")
        st.dataframe(fe.head(8))
        st.markdown("After encoding")
        st.dataframe(enc.head(8))
    notes = []
    if (
        "VisitorType_Returning_Visitor" in enc
        and (enc["IsReturning"] == enc["VisitorType_Returning_Visitor"]).all()
    ):
        notes.append(
            "`IsReturning` (step 06) and `VisitorType_Returning_Visitor` hold identical values"
        )
    if {"Month_Nov", "Month_Dec"} <= set(enc.columns) and (
        enc["IsHolidaySeason"] == enc["Month_Nov"] + enc["Month_Dec"]
    ).all():
        notes.append("`IsHolidaySeason` equals `Month_Nov + Month_Dec`")
    if notes:
        st.caption(
            "Note: "
            + " and ".join(notes)
            + ". They are kept, as in the original implementation."
        )

def page_scaling(raw, D):
    st.header("Feature Scaling")
    X_train = D["X_train"]
    scaler = StandardScaler().fit(X_train)
    scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    sd = X_train.std()
    sd = sd[sd > 0]
    four_part(
        "Standardising each column to mean 0 and standard deviation 1: (value − mean) / std.",
        "Columns use very different units (seconds vs ratios vs 0/1 flags). Models based on distances or margins would be dominated by the columns with the biggest numbers.",
        "All columns get an equal say. The scaler learns mean and std from the **training** rows only, so no test information leaks in.",
        f"In the training data **{sd.idxmax()}** has a standard deviation of {sd.max():,.1f} while **{sd.idxmin()}** has {sd.min():.3f}, "
        f"about {sd.max()/sd.min():,.0f} times smaller. After scaling every column has std 1.",
    )
    st.subheader("Which models use it?")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": mp.MODEL_NAMES,
                "StandardScaler": [
                    "Yes" if MODEL_INFO[n]["scaled"] else "No (not needed)"
                    for n in mp.MODEL_NAMES
                ],
                "Reason": [
                    "Regularised linear model, so scale affects the weights",
                    "Splits on thresholds, so scale does not matter",
                    "Uses distances between visits",
                    "Applied as in the original notebook (Naive Bayes is largely scale-insensitive)",
                    "Margin and RBF kernel use distances",
                ],
            }
        ),
        hide_index=True,
    )
    st.subheader("Before and after")
    default = [
        c
        for c in ["PageValues", "ProductRelated_Duration", "ExitRates", "TotalPages"]
        if c in X_train.columns
    ]
    cols = st.multiselect("Columns to compare", list(X_train.columns), default=default)
    if cols:
        a, b = st.columns(2)
        stat = lambda d: d[cols].agg(["mean", "std", "min", "max"]).T.round(3)
        a.markdown("**Before scaling**")
        a.dataframe(stat(X_train))
        b.markdown("**After scaling**")
        b.dataframe(stat(scaled))
        one = st.selectbox("Distribution of one column", cols)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Before", "After"))
        fig.add_trace(
            go.Histogram(x=X_train[one], nbinsx=40, marker_color="#8a94a6"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=scaled[one], nbinsx=40, marker_color="#e8590c"), row=1, col=2
        )
        fig.update_layout(showlegend=False, height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig)
        st.caption("The shape stays the same. Only the units change.")

def page_train(raw, D):
    st.header("Model Selection and Training")
    names = mp.MODEL_NAMES
    cur = selected_model()
    name = st.selectbox(
        "Choose a classification algorithm", names, index=names.index(cur)
    )
    st.session_state["sel"] = name
    info = MODEL_INFO[name]
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**How it works**")
        st.write(info["principle"])
    with b.container(border=True):
        st.markdown("**How it tackles purchase prediction**")
        st.write(info["approach"])
    st.markdown(f"**Setup in this project:** `{info['settings']}`")
    c = st.columns(4)
    c[0].metric("Training rows", f"{len(D['X_train']):,}")
    c[1].metric("Test rows", f"{len(D['X_test']):,}")
    c[2].metric("Buyers in train", pct(D["y_train"].mean()))
    c[3].metric("Buyers in test", pct(D["y_test"].mean()))
    st.caption(
        "80% train / 20% test, stratified so both parts have the same share of buyers (random_state = 42). "
        "Every model uses exactly the same split. The test rows are never used for training."
    )
    l, r = st.columns([1, 3])
    if l.button(f"Train {name}", type="primary"):
        with st.spinner(f"Training {name}..."):
            train_model(name, D)
        st.success(
            f"{name} trained in {results_store()[name]['seconds']:.1f} s and tested on {len(D['X_test']):,} unseen visits."
        )
    store = results_store()
    if name in store:
        st.markdown("**Test-set scores**")
        metric_row(store[name]["res"]["metrics"])
        st.caption("Go to step 10 for the full evaluation.")
    if store:
        st.markdown(f"Trained so far: {', '.join(n for n in names if n in store)}")
        if r.button("Clear all trained models"):
            store.clear()
            st.rerun()

def metric_story(name, res, D):
    m, cm = res["metrics"], res["cm"]
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    base = 1 - D["y_test"].mean()
    return {
        "Accuracy": f"{m['Accuracy']:.1%} of the {cm.sum():,} test visits were classified correctly. Always predicting 'no purchase' would score {base:.1%}"
        f"{', so accuracy adds little here.' if m['Accuracy'] - base < 0.03 else ', so this is a real gain over that baseline.'}",
        "AUC": f"A random buyer is ranked above a random non-buyer {m['AUC']:.1%} of the time: {band(m['AUC'], [0.9, 0.8, 0.7], ['excellent', 'good', 'fair', 'poor'])} ranking.",
        "Precision": f"Of {tp+fp} visitors flagged as buyers, {tp} really bought ({m['Precision']:.1%}). {fp} were false alarms.",
        "Recall": f"Of {tp+fn} real buyers, {tp} were found ({m['Recall']:.1%}). {fn} sales were missed.",
        "F1": f"Balance of precision and recall: {m['F1']:.3f} ({band(m['F1'], [0.7, 0.5, 0.3], ['strong', 'moderate', 'weak', 'very weak'])}).",
        "MCC": f"{m['MCC']:.3f}: {band(m['MCC'], [0.6, 0.4, 0.2], ['strong', 'moderate', 'weak', 'very weak'])} agreement between predictions and reality, counting all four confusion-matrix cells.",
    }

def page_eval(raw, D):
    st.header("Selected Model Evaluation")
    name = selected_model()
    st.markdown(f"Selected model: **{name}** (change it on step 09)")
    if not need_trained(name, D, "train_eval"):
        return
    res = results_store()[name]["res"]
    m, cm = res["metrics"], res["cm"]
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    metric_row(m)
    st.subheader("What the scores mean")
    story = metric_story(name, res, D)
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": mp.METRICS,
                "Value": [round(m[k], 4) for k in mp.METRICS],
                "Definition": [METRIC_INFO[k] for k in mp.METRICS],
                "For this model": [story[k] for k in mp.METRICS],
            }
        ),
        hide_index=True,
    )
    a, b = st.columns(2)
    with a:
        st.subheader("Confusion matrix")
        st.plotly_chart(cm_fig(cm))
        st.caption(
            f"{tn:,} correct non-buyers · {fp:,} false alarms · {fn:,} missed buyers · {tp:,} correct buyers"
        )
    with b:
        st.subheader("ROC curve")
        st.plotly_chart(roc_fig({name: (res["fpr"], res["tpr"])}, {name: m["AUC"]}))
    st.subheader("Interpretation")
    kind = (
        "catches most buyers but raises many false alarms"
        if m["Recall"] - m["Precision"] > 0.15
        else (
            "is cautious: its buyer flags are usually right but it misses many buyers"
            if m["Precision"] - m["Recall"] > 0.15
            else "keeps precision and recall fairly balanced"
        )
    )
    st.markdown(
        f"**{name}** {kind} (precision {m['Precision']:.2f}, recall {m['Recall']:.2f}). "
        f"Overall ranking quality is {band(m['AUC'], [0.9, 0.8, 0.7], ['excellent', 'good', 'fair', 'poor'])} (AUC {m['AUC']:.3f}) and the "
        f"MCC of {m['MCC']:.3f} is {band(m['MCC'], [0.6, 0.4, 0.2], ['strong', 'moderate', 'weak', 'very weak'])}. "
        f"In business terms, acting on its flags would reach {tp} of {tp+fn} real buyers while also contacting {fp} visitors who would not have bought."
    )
    st.subheader("Actual vs predicted")
    Xt = D["X_test"]
    table = pd.DataFrame(
        {
            "Actual": D["y_test"].map({0: LABELS[0], 1: LABELS[1]}),
            "Predicted": pd.Series(res["pred"], index=Xt.index).map(
                {0: LABELS[0], 1: LABELS[1]}
            ),
            "Score": np.round(res["score"], 3),
        }
    )
    table["Result"] = np.where(
        table["Actual"] == table["Predicted"],
        "Correct",
        np.where(table["Actual"] == LABELS[1], "Missed buyer", "False alarm"),
    )
    table = table.join(Xt[["PageValues", "ProductRelated", "ExitRates"]].round(3))
    flt = st.radio(
        "Show", ["All", "Correct", "Missed buyer", "False alarm"], horizontal=True
    )
    view = table if flt == "All" else table[table["Result"] == flt]
    st.caption(
        f"{len(view):,} rows. Score is the predicted probability (or the decision value for SVM)."
    )
    st.dataframe(view.head(300))

def page_compare(raw, D):
    st.header("Model Comparison")
    store = results_store()
    sel = selected_model()
    others = [n for n in mp.MODEL_NAMES if n not in store]
    if not need_trained(sel, D, "train_cmp"):
        return
    if others:
        st.info(
            "Add models to compare with. Every model uses the same training and test data."
        )
        pick = st.multiselect("Models to train", others, default=others)
        if st.button("Train selected models", type="primary") and pick:
            for n in pick:
                with st.spinner(f"Training {n}..."):
                    train_model(n, D)
            st.rerun()
    rdf = results_df()
    if len(rdf) < 2:
        st.warning("Train at least one more model to compare.")
        return
    best = rdf.idxmax()
    st.subheader("Scores on the test set")
    st.dataframe(rdf.style.highlight_max(axis=0, color="#b2f2bb").format("{:.4f}"))
    st.download_button(
        "Download table (CSV)",
        rdf.round(4).to_csv().encode(),
        "model_comparison.csv",
        "text/csv",
    )
    st.subheader("Best model per metric")
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": mp.METRICS,
                "Best model": [best[k] for k in mp.METRICS],
                "Score": [round(rdf[k].max(), 4) for k in mp.METRICS],
            }
        ),
        hide_index=True,
    )
    t = st.tabs(["Bar chart", "Radar", "ROC curves", "Confusion matrices"])
    with t[0]:
        long = (
            rdf.reset_index()
            .melt(id_vars="index", var_name="Metric", value_name="Score")
            .rename(columns={"index": "Model"})
        )
        fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group")
        fig.update_layout(height=440, margin=dict(t=10))
        st.plotly_chart(fig)
    with t[1]:
        fig = go.Figure()
        for n in rdf.index:
            v = rdf.loc[n, mp.METRICS].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=v + v[:1],
                    theta=mp.METRICS + mp.METRICS[:1],
                    name=n,
                    fill="toself",
                    opacity=0.45,
                )
            )
        fig.update_layout(
            height=480, polar=dict(radialaxis=dict(range=[0, 1])), margin=dict(t=10)
        )
        st.plotly_chart(fig)
    with t[2]:
        rs = {n: store[n]["res"] for n in rdf.index}
        st.plotly_chart(
            roc_fig(
                {n: (r["fpr"], r["tpr"]) for n, r in rs.items()},
                {n: r["metrics"]["AUC"] for n, r in rs.items()},
                500,
            )
        )
    with t[3]:
        cols = st.columns(3)
        for i, n in enumerate(rdf.index):
            cols[i % 3].plotly_chart(
                cm_fig(store[n]["res"]["cm"], n, 300), key=f"cm_{n}"
            )
    st.subheader("Why do the results differ?")
    hi_rec, lo_prec = rdf["Recall"].idxmax(), rdf["Precision"].idxmin()
    spread = rdf["F1"].max() - rdf["F1"].min()
    lines = [
        f"- **F1** ranges from {rdf['F1'].min():.3f} ({rdf['F1'].idxmin()}) to {rdf['F1'].max():.3f} ({rdf['F1'].idxmax()}), a spread of {spread:.3f}.",
        f"- **{hi_rec}** finds the most buyers (recall {rdf.loc[hi_rec, 'Recall']:.3f}), while **{rdf['Precision'].idxmax()}** has the cleanest buyer flags "
        f"(precision {rdf['Precision'].max():.3f})."
        + (
            f" {hi_rec} pays for its recall with the lowest precision."
            if hi_rec == lo_prec
            else ""
        ),
        f"- Accuracy differs by only {rdf['Accuracy'].max() - rdf['Accuracy'].min():.3f} across models, but MCC differs by {rdf['MCC'].max() - rdf['MCC'].min():.3f}. "
        "With few buyers, accuracy hides differences that MCC and F1 reveal.",
        "- Models differ in how they draw the boundary (straight line, rules, neighbours, probabilities, curved margin), in how they treat correlated and skewed columns, "
        "and in the 0.5 cut-off all of them use by default.",
    ]
    st.markdown("\n".join(lines))

def conclusion_for(sel, rdf):
    n = len(rdf)
    ranks = rdf.rank(ascending=False, method="min")
    head = ["F1", "MCC", "AUC"]
    comp = ranks[head].mean(axis=1)
    position = int(comp.rank(method="min")[sel])
    m = rdf.loc[sel]
    best_f1 = rdf["F1"].idxmax()
    strong = [k for k in mp.METRICS if ranks.loc[sel, k] <= 2]
    weak = (
        [k for k in mp.METRICS if ranks.loc[sel, k] >= n - 1]
        if n >= 4
        else [k for k in mp.METRICS if ranks.loc[sel, k] == n]
    )
    info = MODEL_INFO[sel]
    out = {}
    out["overall"] = (
        f"Across the {n} trained models, **{sel}** is #{position} on the three headline measures for imbalanced data "
        f"(F1 {m['F1']:.3f}, MCC {m['MCC']:.3f}, AUC {m['AUC']:.3f}; average rank {comp[sel]:.1f}). "
        f"Its accuracy is {m['Accuracy']:.1%}."
    )
    out["strong"] = (
        ", ".join(f"{k} ({m[k]:.3f}, rank {int(ranks.loc[sel, k])})" for k in strong)
        or "none are in the top two"
    )
    out["weak"] = (
        ", ".join(
            f"{k} ({m[k]:.3f}, rank {int(ranks.loc[sel, k])})"
            for k in weak
            if k not in strong
        )
        or "none are in the bottom two"
    )
    gap = rdf.loc[best_f1, "F1"] - m["F1"]
    out["compare"] = (
        f"{sel} has the best F1 of the group."
        if best_f1 == sel
        else f"{sel}'s F1 is {gap:.3f} below the best model, **{best_f1}** ({rdf.loc[best_f1, 'F1']:.3f})."
    )
    if m["Recall"] - m["Precision"] > 0.15:
        out["pattern"] = (
            f"It leans towards **catching buyers** (recall {m['Recall']:.2f} vs precision {m['Precision']:.2f}), at the cost of false alarms."
        )
    elif m["Precision"] - m["Recall"] > 0.15:
        out["pattern"] = (
            f"It is **cautious** (precision {m['Precision']:.2f} vs recall {m['Recall']:.2f}): flags are usually right but many buyers are missed."
        )
    else:
        out["pattern"] = (
            f"Precision ({m['Precision']:.2f}) and recall ({m['Recall']:.2f}) are fairly balanced."
        )
    rec_best = comp.sort_values().index[0]
    top2 = comp.sort_values().index[:2]
    f1_gap = abs(rdf.loc[top2[0], "F1"] - rdf.loc[top2[1], "F1"])
    mcc_gap = abs(rdf.loc[top2[0], "MCC"] - rdf.loc[top2[1], "MCC"])
    out["winner"] = rec_best
    out["close"] = bool(f1_gap < 0.01 and mcc_gap < 0.01)
    out["info"] = info
    return out

def page_conclusion(raw, D):
    st.header("Conclusion and Recommendation")
    store = results_store()
    sel = selected_model()
    if not need_trained(sel, D, "train_concl"):
        return
    rdf = results_df()
    if len(rdf) < 3:
        st.info(
            "Train more models on step 11 for a meaningful comparison. The conclusion below uses only the trained models."
        )
        others = [n for n in mp.MODEL_NAMES if n not in store]
        if st.button("Train all remaining models", type="primary"):
            for n in others:
                with st.spinner(f"Training {n}..."):
                    train_model(n, D)
            st.rerun()
    if len(rdf) < 2:
        return
    c = conclusion_for(sel, rdf)
    info = c["info"]
    st.subheader(f"Conclusion for {sel}")
    st.markdown(c["overall"])
    a, b = st.columns(2)
    a.success(f"**Strong on:** {c['strong']}")
    b.warning(f"**Weak on:** {c['weak']}")
    st.markdown(f"**Versus the others.** {c['compare']} {c['pattern']}")
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**Strengths of this algorithm**")
        st.markdown("\n".join(f"- {s}" for s in info["strengths"]))
        st.markdown(f"**Why choose it:** {info['choose']}")
    with b.container(border=True):
        st.markdown("**Limitations of this algorithm**")
        st.markdown("\n".join(f"- {s}" for s in info["limits"]))
        st.markdown(f"**When it may be the wrong choice:** {info['avoid']}")
    st.subheader("Recommendation")
    w = c["winner"]
    r = rdf.loc[w]
    n_models = len(rdf)
    st.markdown(
        f"Among the **{n_models}** trained models, **{w}** is the most suitable for predicting purchase intention on this data. "
        f"It has the best average rank on F1, MCC and AUC (F1 {r['F1']:.3f}, MCC {r['MCC']:.3f}, AUC {r['AUC']:.3f}), "
        f"with precision {r['Precision']:.2f} and recall {r['Recall']:.2f}."
    )
    if w != sel:
        st.markdown(
            f"This is not the model you selected. **{sel}** would make sense if: {c['info']['choose'].lower()}"
        )
    if c["close"]:
        st.info(
            "The top two models are within 0.01 on both F1 and MCC. On a single test split that is too close to call, so treat them as roughly equal "
            "and let explainability, speed or the cost of false alarms decide."
        )
    if n_models < 5:
        st.caption(
            f"Only {n_models} of 5 models are trained, so this recommendation may change once all five are included."
        )
    st.warning(
        "Caveats: `PageValues` is calculated from transaction outcomes, so it partly 'knows' the answer and probably lifts every model. "
        "Results come from one 80/20 split, and a different split could move scores slightly."
    )

@st.cache_resource(show_spinner=False)
def load_saved():
    cols_file = MODEL_DIR / "feature_columns.json"
    if not cols_file.exists():
        return dict(
            error="Saved models not found. Run `python train_models.py` and commit the `model/` folder."
        )
    try:
        cols = json.loads(cols_file.read_text())
        models = {
            n: joblib.load(MODEL_DIR / f"{n.lower().replace(' ', '_')}.joblib")
            for n in mp.MODEL_NAMES
        }
        return dict(cols=cols, models=models, error=None)
    except Exception as e:  # e.g. scikit-learn version mismatch
        return dict(
            error=f"Could not load saved models ({type(e).__name__}: {e}). Re-run `python train_models.py` with the same scikit-learn version as requirements.txt."
        )

def page_saved(raw, D):
    st.header("Test the saved models on a CSV")
    st.markdown(
        "This page loads the **already trained** models from the `model/` folder and evaluates them on the CSV you upload. "
        "**Nothing is retrained.** The CSV needs the target column `Revenue` plus either the original dataset columns or the processed feature columns."
    )
    saved = load_saved()
    if saved["error"]:
        st.error(saved["error"])
        return
    up = st.file_uploader("Upload test CSV", type="csv", key="test_upload")
    if up is not None:
        data = read_csv_bytes(up.getvalue())
        st.caption(f"Using uploaded file ({len(data):,} rows).")
    elif (ROOT / "test_data.csv").exists():
        data = read_csv_path(str(ROOT / "test_data.csv"))
        st.caption(
            f"No file uploaded, so the bundled `test_data.csv` is used ({len(data):,} rows)."
        )
    else:
        st.info("Upload a CSV to continue.")
        return
    if mp.TARGET not in data.columns:
        st.error(f"The CSV must contain the target column `{mp.TARGET}`.")
        return
    try:
        X, y = mp.prepare_for_saved_models(data, saved["cols"])
    except Exception as e:
        st.error(str(e))
        return
    if y.nunique() < 2:
        st.error(
            "The target column needs both classes (0 and 1) to compute AUC and MCC."
        )
        return
    with st.expander("Preview of the uploaded data"):
        st.dataframe(data.head(20))
    allres = {n: mp.evaluate(mdl, X, y) for n, mdl in saved["models"].items()}
    table = pd.DataFrame({n: r["metrics"] for n, r in allres.items()}).T[mp.METRICS]
    st.subheader("All five saved models on this data")
    st.dataframe(table.style.highlight_max(axis=0, color="#b2f2bb").format("{:.4f}"))
    long = (
        table.reset_index()
        .melt(id_vars="index", var_name="Metric", value_name="Score")
        .rename(columns={"index": "Model"})
    )
    st.plotly_chart(
        px.bar(
            long, x="Metric", y="Score", color="Model", barmode="group"
        ).update_layout(height=400, margin=dict(t=10))
    )
    st.subheader("Inspect one model")
    name = st.selectbox("Model", mp.MODEL_NAMES, key="saved_model_pick")
    r = allres[name]
    metric_row(r["metrics"])
    a, b = st.columns(2)
    a.plotly_chart(cm_fig(r["cm"], "Confusion matrix"), key="saved_cm")
    b.plotly_chart(
        roc_fig({name: (r["fpr"], r["tpr"])}, {name: r["metrics"]["AUC"]}),
        key="saved_roc",
    )
    st.markdown("**Classification report**")
    rep = pd.DataFrame(
        classification_report(
            y, r["pred"], target_names=LABELS, output_dict=True, zero_division=0
        )
    ).T.round(3)
    st.dataframe(rep)

PAGE_FUNCS = [
    page_intro,
    page_dataset,
    page_eda,
    page_cleaning,
    page_corr,
    page_fe,
    page_encoding,
    page_scaling,
    page_train,
    page_eval,
    page_compare,
    page_conclusion,
    page_saved,
]

# ====================================================================== main
def main():
    with st.sidebar:
        st.title("🛒 Shopper App")
        st.radio("Steps", PAGES, key="nav", label_visibility="collapsed")
        trained = list(results_store())
        st.divider()
        st.caption("Selected model: **" + selected_model() + "**")
        st.caption(
            "Trained: "
            + (", ".join(n for n in mp.MODEL_NAMES if n in trained) or "none yet")
        )
    raw = load_raw()
    D = build_data(raw)
    i = PAGES.index(st.session_state.get("nav", PAGES[0]))
    PAGE_FUNCS[i](raw, D)
    nav_buttons(i)

main()
