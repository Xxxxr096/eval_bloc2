# app.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartMarket - Marketing Dashboard", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def pct(x):
    return f"{x*100:.2f}%"


def eur(x):
    return f"{x:,.2f} €".replace(",", " ").replace(".", ",")


def safe_div(a, b):
    return np.nan if b in (0, 0.0, None) or pd.isna(b) else a / b


def load_or_fallback():
    """
    Try loading the 3 files from the working directory:
      - leads_smartmarket.csv
      - crm_smartmarket.xlsx
      - campaign_smartmarket.json
    If any are missing, use the dataset as given in the prompt (fallback).
    """
    leads_path = Path("leads_smartmarket.csv")
    crm_path = Path("crm_smartmarket.xlsx")
    camp_path = Path("campaign_smartmarket.json")

    # cas 1 : on a bien les fichiers
    if leads_path.exists() and crm_path.exists() and camp_path.exists():
        leads = pd.read_csv(leads_path)
        crm = pd.read_excel(crm_path)
        with open(camp_path, "r", encoding="utf-8") as f:
            campaigns = pd.DataFrame(json.load(f))
        source = "Fichiers locaux chargés"
    else:
        # cas 2 : on les a pas
        leads = pd.DataFrame(
            [
                {
                    "lead_id": 101,
                    "date": "2025-09-01",
                    "channel": "Emailing",
                    "device": "Mobile",
                },
                {
                    "lead_id": 102,
                    "date": "2025-09-02",
                    "channel": "Facebook Ads",
                    "device": "Desktop",
                },
                {
                    "lead_id": 103,
                    "date": "2025-09-03",
                    "channel": "LinkedIn",
                    "device": "Mobile",
                },
                {
                    "lead_id": 104,
                    "date": "2025-09-04",
                    "channel": "Emailing",
                    "device": "Tablet",
                },
                {
                    "lead_id": 105,
                    "date": "2025-09-05",
                    "channel": "Instagram Ads",
                    "device": "Mobile",
                },
            ]
        )
        crm = pd.DataFrame(
            [
                {
                    "lead_id": 101,
                    "company_size": "1-10",
                    "sector": "Tech",
                    "region": "IdF",
                    "status": "MQL",
                },
                {
                    "lead_id": 102,
                    "company_size": "10-50",
                    "sector": "Retail",
                    "region": "Hauts-de-France",
                    "status": "SQL",
                },
                {
                    "lead_id": 103,
                    "company_size": "50-100",
                    "sector": "Finance",
                    "region": "PAC",
                    "status": "Client",
                },
                {
                    "lead_id": 104,
                    "company_size": "1-10",
                    "sector": "Health",
                    "region": "ARA",
                    "status": "MQL",
                },
                {
                    "lead_id": 105,
                    "company_size": "100-500",
                    "sector": "Education",
                    "region": "IdF",
                    "status": "Client",
                },
            ]
        )
        campaigns = pd.DataFrame(
            [
                {
                    "campaign_id": "CAMP01",
                    "channel": "Emailing",
                    "cost": 1200,
                    "impressions": 50000,
                    "clicks": 1500,
                    "conversions": 120,
                },
                {
                    "campaign_id": "CAMP02",
                    "channel": "Facebook Ads",
                    "cost": 3000,
                    "impressions": 90000,
                    "clicks": 2200,
                    "conversions": 180,
                },
                {
                    "campaign_id": "CAMP03",
                    "channel": "LinkedIn",
                    "cost": 2500,
                    "impressions": 40000,
                    "clicks": 900,
                    "conversions": 75,
                },
                {
                    "campaign_id": "CAMP04",
                    "channel": "Instagram Ads",
                    "cost": 1800,
                    "impressions": 70000,
                    "clicks": 2600,
                    "conversions": 210,
                },
            ]
        )
        source = "Fallback (données de l’énoncé)"

    # Petit nettoyages
    leads["date"] = pd.to_datetime(leads["date"], errors="coerce")
    leads["lead_id"] = pd.to_numeric(leads["lead_id"], errors="coerce").astype("Int64")
    crm["lead_id"] = pd.to_numeric(crm["lead_id"], errors="coerce").astype("Int64")

    for col in ["cost", "impressions", "clicks", "conversions"]:
        campaigns[col] = pd.to_numeric(campaigns[col], errors="coerce")

    return leads, crm, campaigns, source


def compute_campaign_kpis(campaigns: pd.DataFrame) -> pd.DataFrame:
    df = campaigns.copy()
    df["ctr"] = df.apply(lambda r: safe_div(r["clicks"], r["impressions"]), axis=1)
    df["conv_rate_click"] = df.apply(
        lambda r: safe_div(r["conversions"], r["clicks"]), axis=1
    )
    df["cpc"] = df.apply(lambda r: safe_div(r["cost"], r["clicks"]), axis=1)
    df["cpa"] = df.apply(lambda r: safe_div(r["cost"], r["conversions"]), axis=1)
    return df


def plot_bar(x, y, title, ylabel, rotate=25):
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(x, rotation=rotate, ha="right")
    fig.tight_layout()
    return fig


# Chargement des données
leads, crm, campaigns, source = load_or_fallback()
st.caption(f"Source des données : **{source}**")

# Jointure leads + CRM sur lead_id
df_leads = leads.merge(crm, on="lead_id", how="left", validate="one_to_one")

# Filtrage du périmetre
period_start = pd.Timestamp("2025-09-01")
period_end = pd.Timestamp("2025-09-30")
df_leads = df_leads[
    (df_leads["date"] >= period_start) & (df_leads["date"] <= period_end)
].copy()

# Calcule des KPI campagnes
df_camp = compute_campaign_kpis(campaigns)

# Filtres
st.sidebar.header("Filtres")

# On propose des filtre simple
all_channels = sorted(df_leads["channel"].dropna().unique().tolist())
sel_channels = st.sidebar.multiselect("Canaux", all_channels, default=all_channels)

all_regions = sorted(df_leads["region"].dropna().unique().tolist())
sel_regions = st.sidebar.multiselect("Régions", all_regions, default=all_regions)

all_status = sorted(df_leads["status"].dropna().unique().tolist())
sel_status = st.sidebar.multiselect("Statuts", all_status, default=all_status)

all_devices = sorted(df_leads["device"].dropna().unique().tolist())
sel_devices = st.sidebar.multiselect("Devices", all_devices, default=all_devices)

# Application des filtre sur les leads + CRM
df_leads_f = df_leads[
    df_leads["channel"].isin(sel_channels)
    & df_leads["region"].isin(sel_regions)
    & df_leads["status"].isin(sel_status)
    & df_leads["device"].isin(sel_devices)
].copy()

# Les compagne sont filtrer au minimum par canal
df_camp_f = df_camp[df_camp["channel"].isin(sel_channels)].copy()

# Calcule du nombre de leads par canal
leads_by_channel = (
    df_leads_f.groupby("channel", dropna=False)["lead_id"]
    .nunique()
    .rename("leads")
    .reset_index()
)
df_camp_f = df_camp_f.merge(
    leads_by_channel, left_on="channel", right_on="channel", how="left"
)
df_camp_f["cpl_indicatif"] = df_camp_f.apply(
    lambda r: safe_div(r["cost"], r["leads"]), axis=1
)

# Titre
st.title("SmartMarket — Dashboard Marketing (Septembre 2025)")
st.write(
    "Tableau de bord de synthèse : performance par canal (CTR, conversion, coûts) "
    "et lecture qualitative via les statuts CRM / régions."
)

# KPI globaux
total_cost = df_camp_f["cost"].sum()
total_impr = df_camp_f["impressions"].sum()
total_clicks = df_camp_f["clicks"].sum()
total_conv = df_camp_f["conversions"].sum()

ctr_global = safe_div(total_clicks, total_impr)
conv_rate_global = safe_div(total_conv, total_clicks)
cpa_global = safe_div(total_cost, total_conv)

total_leads = df_leads_f["lead_id"].nunique()
cpl_global_indicatif = safe_div(total_cost, total_leads)

# Ligne de 6 cartes KPI
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Budget total", eur(total_cost))
k2.metric("Leads (périmètre)", f"{total_leads}")
k3.metric("CTR global", "—" if pd.isna(ctr_global) else pct(ctr_global))
k4.metric(
    "Taux conv (clic→conv)", "—" if pd.isna(conv_rate_global) else pct(conv_rate_global)
)
k5.metric("CPA global", "—" if pd.isna(cpa_global) else eur(cpa_global))
k6.metric(
    "CPL (indicatif)",
    "—" if pd.isna(cpl_global_indicatif) else eur(cpl_global_indicatif),
)

st.caption(
    "Note : le CPL est **indicatif** car les conversions/clicks/impressions sont agrégés par canal, "
    "et le lien direct lead→campagne n’est pas fourni."
)

st.divider()

# Graphique principaux
left, right = st.columns(2)


with left:
    st.subheader("CTR par canal")
    tmp = df_camp_f[["channel", "ctr"]].sort_values("ctr", ascending=False)
    fig = plot_bar(tmp["channel"], tmp["ctr"], "Taux de clic (CTR) par canal", "CTR")
    st.pyplot(fig, clear_figure=True)
    st.write("Lecture : comparaison de l’engagement (clics/impressions) entre canaux.")


with right:
    st.subheader("CPA par canal")
    tmp = df_camp_f[["channel", "cpa"]].sort_values("cpa", ascending=True)
    fig = plot_bar(
        tmp["channel"], tmp["cpa"], "Coût par conversion (CPA) par canal", "CPA (€)"
    )
    st.pyplot(fig, clear_figure=True)
    st.write(
        "Lecture : comparaison de la rentabilité (coût par conversion) entre canaux."
    )

st.divider()

left2, right2 = st.columns(2)


with left2:
    st.subheader("Qualité des leads : statut par canal")
    pivot = df_leads_f.pivot_table(
        index="channel",
        columns="status",
        values="lead_id",
        aggfunc="nunique",
        fill_value=0,
    ).reindex(sel_channels)
    fig, ax = plt.subplots()
    bottoms = np.zeros(len(pivot.index))
    for col in pivot.columns:
        ax.bar(pivot.index, pivot[col].values, bottom=bottoms, label=str(col))
        bottoms += pivot[col].values
    ax.set_title("Répartition des leads par statut et par canal")
    ax.set_ylabel("Nombre de leads")
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.write(
        "Lecture : arbitrage volume vs maturité (MQL/SQL/Client) selon les canaux."
    )


with right2:
    st.subheader("Répartition géographique")
    reg = df_leads_f.groupby("region")["lead_id"].nunique().sort_values(ascending=False)
    fig = plot_bar(
        reg.index.tolist(),
        reg.values.tolist(),
        "Répartition des leads par région",
        "Nombre de leads",
        rotate=0,
    )
    st.pyplot(fig, clear_figure=True)
    st.write(
        "Lecture : identification des zones les plus contributrices en volume de leads."
    )

st.divider()

# Table de détail

with st.expander("Détails KPI par canal (table)"):
    out = df_camp_f.copy()
    out = out[
        [
            "channel",
            "cost",
            "impressions",
            "clicks",
            "conversions",
            "ctr",
            "conv_rate_click",
            "cpc",
            "cpa",
            "leads",
            "cpl_indicatif",
        ]
    ]
    out = out.sort_values("cpa")
    st.dataframe(out, use_container_width=True)

with st.expander("Données leads + CRM (après filtres)"):
    st.dataframe(df_leads_f.sort_values(["date", "lead_id"]), use_container_width=True)

st.caption("SmartMarket — Dashboard (Streamlit).")
