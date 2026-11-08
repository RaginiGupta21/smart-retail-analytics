
from pathlib import Path
from importlib.metadata import version as pkg_version

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Parameters copied from the notebook (do not change without changing the notebook)
# ----------------------------------------------------------------------------
DATA_FILE = "online_retail_uk.csv"
REQUIRED_COLUMNS = ["InvoiceNo", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]

# K-Means (cell 12)
K_RANGE = range(2, 9)
K_FINAL = 4
RANDOM_STATE = 42
N_INIT = 10
NAME_MAP = {  # cluster id -> segment name, exactly as in the notebook
    3: "Champions",
    1: "Loyal Customers",
    2: "Regular Customers",
    0: "At-Risk / Hibernating",
}
EXPECTED_ORDER_BY_SPEND = ["Champions", "Loyal Customers", "Regular Customers", "At-Risk / Hibernating"]

# Apriori - overall (cell 17)
OVERALL_TOP_ITEMS = 100
OVERALL_MIN_SUPPORT = 0.02
OVERALL_MIN_LIFT = 1.0

# Apriori - per segment (cell 18)
SEG_TOP_ITEMS = 80
SEG_MIN_SUPPORT = 0.03
SEG_MIN_LIFT = 1.2
MIN_CUSTOMERS_FOR_RELIABLE_APRIORI = 10
GLOBAL_TOP_PAIRS_N = 15

# Business impact (cell 19)
TEACUPS = [
    "PINK REGENCY TEACUP AND SAUCER",
    "GREEN REGENCY TEACUP AND SAUCER",
    "ROSES REGENCY TEACUP AND SAUCER",
]
CONVERSION_ASSUMPTION = 0.25
WINBACK_CONVERSION = 0.20

# Soft, muted palette
SEG_COLORS = {
    "Champions": "#C8A951",
    "Loyal Customers": "#5B8DB8",
    "Regular Customers": "#8FAE9A",
    "At-Risk / Hibernating": "#C27C7C",
}
NAVY = "#1F3A5F"
STEEL = "#5B7DB1"
SAGE = "#8FAE9A"

GITHUB_URL = "https://github.com/RaginiGupta21/smart-retail-analytics"
LINKEDIN_URL = "https://www.linkedin.com/in/raginigupta21"

st.set_page_config(
    page_title="Smart Retail Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      h1 {letter-spacing: -0.5px;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def money(x, decimals=2):
    return f"£{x:,.{decimals}f}"


def find_data_file():
    """Look for data/online_retail_uk.csv next to app.py, one level up, or in the working dir."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "data" / DATA_FILE,
        here.parent / "data" / DATA_FILE,
        Path.cwd() / "data" / DATA_FILE,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_association_rules(freq, n_rows, min_lift):
    """association_rules() exactly as in the notebook (lift metric). Newer mlxtend versions
    accept an extra num_itemsets argument; it does not change the lift/support/confidence values."""
    if freq.empty:
        return pd.DataFrame()
    try:
        return association_rules(freq, metric="lift", min_threshold=min_lift)
    except TypeError:
        return association_rules(freq, num_itemsets=n_rows, metric="lift", min_threshold=min_lift)


def join_items(rules):
    rules = rules.copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    return rules


def compute_impact(res, conv, winback_conv):
    """Business impact - same formulas as notebook cell 19."""
    summary = res["summary"]
    potential_customers = res["potential_customers"]
    avg_line_revenue = res["avg_line_revenue"]

    projected_conversions = int(potential_customers * conv)
    projected_revenue_crosssell = projected_conversions * avg_line_revenue

    at_risk = summary[summary["SegmentName"] == "At-Risk / Hibernating"].iloc[0]
    regular = summary[summary["SegmentName"] == "Regular Customers"].iloc[0]
    winback_customers = int(at_risk["Customers"] * winback_conv)
    revenue_uplift_per_customer = regular["AvgMonetary"] - at_risk["AvgMonetary"]
    projected_winback_revenue = winback_customers * revenue_uplift_per_customer

    total_revenue = res["total_revenue"]
    total_impact = projected_revenue_crosssell + projected_winback_revenue
    return {
        "potential_customers": potential_customers,
        "projected_conversions": projected_conversions,
        "avg_line_revenue": avg_line_revenue,
        "projected_revenue_crosssell": projected_revenue_crosssell,
        "at_risk_customers": int(at_risk["Customers"]),
        "winback_customers": winback_customers,
        "revenue_uplift_per_customer": revenue_uplift_per_customer,
        "projected_winback_revenue": projected_winback_revenue,
        "total_revenue": total_revenue,
        "total_impact": total_impact,
        "pct": 100 * total_impact / total_revenue,
    }


# ----------------------------------------------------------------------------
# The full analysis (cached). Mirrors the notebook cell by cell.
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_analysis(csv_path: str, file_signature: tuple) -> dict:
    res = {}

    # ---- Cell 1 + 3: load and inspect raw data -----------------------------
    df = pd.read_csv(csv_path)
    res["raw_rows"], res["raw_cols"] = df.shape
    res["raw_quality"] = {
        "Null values (total)": int(df.isnull().sum().sum()),
        "Null CustomerID": int(df["CustomerID"].isnull().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
        "Cancelled orders (InvoiceNo starts with 'C')": int(df["InvoiceNo"].astype(str).str.startswith("C").sum()),
        "Zero or negative quantity": int((df["Quantity"] <= 0).sum()),
        "Zero or negative price": int((df["UnitPrice"] <= 0).sum()),
    }

    # ---- Cell 5: data cleaning (same order as the notebook) -----------------
    funnel = [("Raw rows", len(df))]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    funnel.append(("After removing cancellations", len(df)))
    df = df.dropna(subset=["CustomerID"])
    funnel.append(("After removing missing CustomerID", len(df)))
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    funnel.append(("After removing zero/negative quantity and price", len(df)))
    df = df.drop_duplicates()
    funnel.append(("After removing duplicates (analysis dataset)", len(df)))
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    res["funnel"] = funnel

    # ---- Cell 7: EDA ------------------------------------------------------
    total_revenue = df["TotalPrice"].sum()
    res["total_revenue"] = total_revenue
    res["n_customers"] = df["CustomerID"].nunique()
    res["n_invoices"] = df["InvoiceNo"].nunique()
    res["n_products"] = df["Description"].nunique()
    res["date_min"] = df["InvoiceDate"].min()
    res["date_max"] = df["InvoiceDate"].max()
    res["top_products"] = (
        df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10)
    )
    res["monthly"] = df.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum()

    # ---- Cell 9: RFM ------------------------------------------------------
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()
    res["snapshot_date"] = snapshot_date

    # ---- Cell 12: K-Means -------------------------------------------------
    X = rfm[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    k_rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        k_rows.append({"k": k, "WCSS (Elbow)": km.inertia_, "Silhouette Score": sil})
    res["k_table"] = pd.DataFrame(k_rows)

    km_final = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=N_INIT).fit(Xs)
    rfm["Cluster"] = km_final.labels_

    summary = rfm.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        TotalMonetary=("Monetary", "sum"),
    ).round(1)
    summary["PctCustomers"] = (100 * summary["Customers"] / summary["Customers"].sum()).round(1)
    summary["PctRevenue"] = (100 * summary["TotalMonetary"] / summary["TotalMonetary"].sum()).round(1)

    rfm["Segment"] = rfm["Cluster"].map(NAME_MAP)
    summary["SegmentName"] = summary.index.map(NAME_MAP)
    summary = summary.sort_values("AvgMonetary", ascending=False)
    res["rfm"] = rfm
    res["summary"] = summary

    # Safety check on the hard-coded cluster-id -> name mapping
    problems = []
    if list(summary["SegmentName"]) != EXPECTED_ORDER_BY_SPEND:
        problems.append(
            "Segments ranked by average spend are not in the order Champions > Loyal > Regular > At-Risk."
        )
    if summary.loc[summary["AvgRecency"].idxmax(), "SegmentName"] != "At-Risk / Hibernating":
        problems.append("'At-Risk / Hibernating' is not the segment with the highest average recency.")
    res["mapping_problems"] = problems

    # ---- Cell 17: Apriori - overall ---------------------------------------
    item_freq = df.groupby("Description")["InvoiceNo"].nunique().sort_values(ascending=False)
    top_items = item_freq.head(OVERALL_TOP_ITEMS).index
    df_top = df[df["Description"].isin(top_items)]
    res["overall_invoices_covered"] = df_top["InvoiceNo"].nunique()

    basket = df_top.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().fillna(0)
    basket_sets = basket > 0
    res["overall_basket_shape"] = basket_sets.shape

    frequent_itemsets = apriori(basket_sets, min_support=OVERALL_MIN_SUPPORT, use_colnames=True)
    rules = run_association_rules(frequent_itemsets, len(basket_sets), OVERALL_MIN_LIFT)
    rules = rules.sort_values("lift", ascending=False)
    rules = join_items(rules)
    res["overall_itemsets"] = len(frequent_itemsets)
    res["overall_rules"] = rules

    # ---- Cell 18: Apriori - per segment (with reliability check) ----------
    df_seg = df.merge(rfm[["CustomerID", "Segment"]], on="CustomerID", how="left")
    global_top_pairs = set(
        zip(rules.head(GLOBAL_TOP_PAIRS_N)["antecedents"], rules.head(GLOBAL_TOP_PAIRS_N)["consequents"])
    )

    seg_status = []
    all_segment_rules = []
    for seg in df_seg["Segment"].unique():
        seg_df = df_seg[df_seg["Segment"] == seg]
        n_customers = seg_df["CustomerID"].nunique()
        n_invoices = seg_df["InvoiceNo"].nunique()

        if n_customers < MIN_CUSTOMERS_FOR_RELIABLE_APRIORI:
            seg_status.append({
                "Segment": seg, "Customers": n_customers, "Invoices": n_invoices,
                "Status": f"Skipped (fewer than {MIN_CUSTOMERS_FOR_RELIABLE_APRIORI} customers)",
                "Rules found": 0,
            })
            continue

        s_freq = seg_df.groupby("Description")["InvoiceNo"].nunique().sort_values(ascending=False)
        s_top = s_freq.head(SEG_TOP_ITEMS).index
        seg_top = seg_df[seg_df["Description"].isin(s_top)]
        s_basket = seg_top.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().fillna(0)
        s_basket_sets = s_basket > 0

        freq_items = apriori(s_basket_sets, min_support=SEG_MIN_SUPPORT, use_colnames=True)
        rules_seg = run_association_rules(freq_items, len(s_basket_sets), SEG_MIN_LIFT)
        if rules_seg.empty:
            seg_status.append({
                "Segment": seg, "Customers": n_customers, "Invoices": n_invoices,
                "Status": "Analysed (no rules passed the thresholds)", "Rules found": 0,
            })
            continue
        rules_seg = rules_seg.sort_values("lift", ascending=False)
        rules_seg = join_items(rules_seg)
        rules_seg["Segment"] = seg
        rules_seg["Pattern"] = [
            "Matches global pattern" if (a, c) in global_top_pairs else "Unique to this segment"
            for a, c in zip(rules_seg["antecedents"], rules_seg["consequents"])
        ]
        all_segment_rules.append(rules_seg)
        seg_status.append({
            "Segment": seg, "Customers": n_customers, "Invoices": n_invoices,
            "Status": "Analysed", "Rules found": len(rules_seg),
        })

    res["seg_status"] = pd.DataFrame(seg_status)
    res["seg_rules"] = pd.concat(all_segment_rules, ignore_index=True) if all_segment_rules else pd.DataFrame()

    # ---- Cell 19: business impact inputs ----------------------------------
    sub = df[df["Description"].isin(TEACUPS)].copy()
    sub["LineRevenue"] = sub["Quantity"] * sub["UnitPrice"]
    res["avg_line_revenue"] = sub["LineRevenue"].mean()

    t_basket = (
        df[df["Description"].isin(TEACUPS)]
        .groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().fillna(0)
    )
    t_basket_bool = t_basket > 0
    exactly_2 = t_basket_bool[t_basket_bool.sum(axis=1) == 2]
    res["potential_customers"] = len(exactly_2)

    # Diagnostic only (not used in any calculation): which teacup names exist in the data?
    res["teacup_match_counts"] = {t: int((df["Description"] == t).sum()) for t in TEACUPS}
    res["teacup_variants"] = sorted(
        {repr(d) for d in df["Description"].dropna().unique() if "REGENCY TEACUP" in d}
    )
    res["teacup_columns_used"] = list(t_basket_bool.columns)

    return res


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
data_path = find_data_file()
if data_path is None:
    st.title("Smart Retail Analytics")
    st.error(
        f"Could not find `data/{DATA_FILE}`.\n\n"
        "Download the UCI **Online Retail** dataset, keep the United Kingdom rows, and save them as "
        f"`data/{DATA_FILE}` (the `data` folder sits next to this app, or one level above it)."
    )
    st.stop()

stat = data_path.stat()
with st.spinner("Running the analysis (cleaning, RFM, K-Means, Apriori). The first run takes a little while..."):
    try:
        R = run_analysis(str(data_path), (stat.st_mtime, stat.st_size))
    except KeyError as e:
        st.error(f"The CSV is missing an expected column: {e}. Expected columns: {', '.join(REQUIRED_COLUMNS)}.")
        st.stop()

rfm = R["rfm"]
summary = R["summary"]
rules_overall = R["overall_rules"]
seg_rules = R["seg_rules"]
seg_status = R["seg_status"]
impact_default = compute_impact(R, CONVERSION_ASSUMPTION, WINBACK_CONVERSION)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Smart Retail Analytics")
    st.caption("Customer segmentation · Market basket analysis · Business impact")
    st.markdown(
        "A Data Mining & Warehousing case study on one year of UK online-retail "
        "transactions (UCI Online Retail dataset)."
    )
    st.divider()
    st.markdown("**Method at a glance**")
    st.markdown(
        f"- RFM features, `StandardScaler`, K-Means (k = {K_FINAL})\n"
        f"- Apriori: top {OVERALL_TOP_ITEMS} products, min support {OVERALL_MIN_SUPPORT}\n"
        f"- Segment-wise Apriori: top {SEG_TOP_ITEMS} products, min support {SEG_MIN_SUPPORT}, "
        f"min lift {SEG_MIN_LIFT}\n"
        f"- Segments with fewer than {MIN_CUSTOMERS_FOR_RELIABLE_APRIORI} customers are skipped"
    )
    st.divider()
    st.markdown(f"[GitHub repository]({GITHUB_URL})  \n[LinkedIn]({LINKEDIN_URL})")
    st.caption("Built by Ragini Gupta")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🛍️ Smart Retail Analytics")
st.caption(
    "RFM + K-Means customer segmentation, Apriori market basket analysis and a projected business impact "
    "for a UK online retailer."
)
if R["mapping_problems"]:
    st.warning(
        "The cluster numbers produced in this environment do not match the segment names assigned in the "
        "notebook (" + " ".join(R["mapping_problems"]) + "). K-Means cluster ids can vary between library "
        "versions, so please verify the segment names before relying on the results."
    )

tab_overview, tab_seg, tab_basket, tab_impact, tab_insights, tab_method = st.tabs(
    ["Overview", "Customer segments", "Market basket", "Business impact", "Insights", "Method & data"]
)

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total revenue", money(R["total_revenue"], 0))
    c2.metric("Customers", f"{R['n_customers']:,}")
    c3.metric("Invoices", f"{R['n_invoices']:,}")
    c4.metric("Products", f"{R['n_products']:,}")
    st.caption(f"Data covers {R['date_min'].date()} to {R['date_max'].date()}. "
               f"All figures are for the cleaned dataset ({R['funnel'][-1][1]:,} rows).")

    left, right = st.columns([3, 2])
    with left:
        monthly = R["monthly"].reset_index()
        monthly.columns = ["Month", "Revenue"]
        fig = px.bar(monthly, x="Month", y="Revenue", title="Monthly revenue",
                     color_discrete_sequence=[STEEL])
        fig.update_traces(hovertemplate="%{x|%b %Y}<br>£%{y:,.0f}<extra></extra>")
        fig.update_layout(yaxis_tickprefix="£", yaxis_title=None, xaxis_title=None, height=380)
        st.plotly_chart(fig, width="stretch")
    with right:
        tp = R["top_products"].reset_index()
        tp.columns = ["Product", "Revenue"]
        tp = tp.iloc[::-1]
        fig = px.bar(tp, x="Revenue", y="Product", orientation="h", title="Top 10 products by revenue",
                     color_discrete_sequence=[NAVY])
        fig.update_traces(hovertemplate="%{y}<br>£%{x:,.0f}<extra></extra>")
        fig.update_layout(xaxis_tickprefix="£", xaxis_title=None, yaxis_title=None, height=380)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Segment snapshot")
    snap = summary[["SegmentName", "Customers", "PctCustomers", "PctRevenue"]].copy()
    cols = st.columns(len(snap))
    for col, (_, row) in zip(cols, snap.iterrows()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{row['SegmentName']}**")
                st.metric("Customers", f"{int(row['Customers']):,}", f"{row['PctCustomers']}% of customers",
                          delta_color="off")
                st.caption(f"{row['PctRevenue']}% of revenue")

    st.subheader("Data cleaning")
    fdf = pd.DataFrame(R["funnel"], columns=["Step", "Rows"])
    fig = px.bar(fdf.iloc[::-1], x="Rows", y="Step", orientation="h", color_discrete_sequence=[SAGE])
    fig.update_traces(hovertemplate="%{y}<br>%{x:,} rows<extra></extra>")
    fig.update_layout(height=300, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# Customer segments
# ----------------------------------------------------------------------------
with tab_seg:
    st.subheader("Segment profiles")
    st.caption(
        "Each customer is described by Recency (days since last purchase), Frequency (number of invoices) and "
        "Monetary value (total spend), scaled with StandardScaler and clustered with K-Means."
    )
    prof = summary[["SegmentName", "Customers", "PctCustomers", "AvgRecency", "AvgFrequency",
                    "AvgMonetary", "TotalMonetary", "PctRevenue"]].rename(columns={
        "SegmentName": "Segment", "PctCustomers": "% customers", "AvgRecency": "Avg recency (days)",
        "AvgFrequency": "Avg frequency", "AvgMonetary": "Avg spend (£)",
        "TotalMonetary": "Total spend (£)", "PctRevenue": "% revenue"})
    st.dataframe(
        prof, hide_index=True, width="stretch",
        column_config={
            "Customers": st.column_config.NumberColumn(format="%d"),
            "Avg spend (£)": st.column_config.NumberColumn(format="£%.2f"),
            "Total spend (£)": st.column_config.NumberColumn(format="£%.0f"),
        },
    )

    a, b = st.columns(2)
    with a:
        share = summary[["SegmentName", "PctCustomers", "PctRevenue"]].melt(
            id_vars="SegmentName", var_name="Measure", value_name="Percent")
        share["Measure"] = share["Measure"].map({"PctCustomers": "% of customers", "PctRevenue": "% of revenue"})
        fig = px.bar(share, x="SegmentName", y="Percent", color="Measure", barmode="group",
                     title="Share of customers vs share of revenue",
                     color_discrete_map={"% of customers": STEEL, "% of revenue": "#C8A951"},
                     category_orders={"SegmentName": list(summary["SegmentName"])})
        fig.update_layout(xaxis_title=None, yaxis_title="%", height=380, legend_title=None)
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.bar(summary, x="SegmentName", y="AvgRecency", color="SegmentName",
                     color_discrete_map=SEG_COLORS, title="Average recency by segment (days)")
        fig.update_layout(xaxis_title=None, yaxis_title="Days", height=380, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Explore customers")
    f1, f2 = st.columns([3, 1])
    chosen = f1.multiselect("Segments to show", list(summary["SegmentName"]),
                            default=list(summary["SegmentName"]))
    log_y = f2.checkbox("Log scale for spend", value=True)
    plot_df = rfm[rfm["Segment"].isin(chosen)]
    fig = px.scatter(plot_df, x="Recency", y="Monetary", color="Segment", color_discrete_map=SEG_COLORS,
                     hover_data={"CustomerID": True, "Frequency": True, "Recency": True, "Monetary": ":,.2f"},
                     log_y=log_y, opacity=0.65, title="Recency vs spend (one dot per customer)")
    fig.update_layout(height=480, xaxis_title="Recency (days since last purchase)",
                      yaxis_title="Total spend (£)", legend_title=None)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Choosing k")
    st.caption(f"K-Means was tested for k = {K_RANGE.start} to {K_RANGE.stop - 1}. k = {K_FINAL} was chosen "
               "because it balances statistical fit with segments a marketer can act on.")
    kt = R["k_table"]
    e1, e2 = st.columns(2)
    with e1:
        fig = px.line(kt, x="k", y="WCSS (Elbow)", markers=True, title="Elbow (WCSS)",
                      color_discrete_sequence=[NAVY])
        fig.add_vline(x=K_FINAL, line_dash="dot", line_color="#C27C7C")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    with e2:
        fig = px.line(kt, x="k", y="Silhouette Score", markers=True, title="Silhouette score",
                      color_discrete_sequence=[STEEL])
        fig.add_vline(x=K_FINAL, line_dash="dot", line_color="#C27C7C")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Customer list")
    st.dataframe(
        rfm.sort_values("Monetary", ascending=False)[["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]],
        hide_index=True, width="stretch", height=320,
        column_config={"Monetary": st.column_config.NumberColumn(format="£%.2f")},
    )

# ----------------------------------------------------------------------------
# Market basket
# ----------------------------------------------------------------------------
with tab_basket:
    with st.expander("How to read support, confidence and lift"):
        st.markdown(
            "- **Support**: how often the combination appears in baskets.\n"
            "- **Confidence**: when A is bought, how often is B also bought?\n"
            "- **Lift**: is A + B happening more than we would expect by chance? "
            "Lift = 1 means about as often as chance, above 1 means more often, below 1 means less often."
        )

    sub_overall, sub_seg = st.tabs(["Overall", "By segment"])

    # ---- overall ----
    with sub_overall:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Invoices covered", f"{R['overall_invoices_covered']:,}")
        m2.metric("Basket matrix", f"{R['overall_basket_shape'][0]:,} × {R['overall_basket_shape'][1]}")
        m3.metric("Frequent itemsets", f"{R['overall_itemsets']:,}")
        m4.metric("Association rules", f"{len(rules_overall):,}")
        st.caption(f"Restricted to the top {OVERALL_TOP_ITEMS} most frequent products. "
                   f"Apriori min support = {OVERALL_MIN_SUPPORT}; rules kept when lift ≥ {OVERALL_MIN_LIFT}.")

        if rules_overall.empty:
            st.info("No association rules were found with these thresholds.")
        else:
            f1, f2, f3 = st.columns([1, 1, 2])
            min_lift = f1.slider("Minimum lift", float(OVERALL_MIN_LIFT),
                                 float(round(rules_overall["lift"].max(), 1)), float(OVERALL_MIN_LIFT), 0.5)
            top_n = f2.slider("Rules in chart", 5, 30, 10)
            query = f3.text_input("Search a product", placeholder="e.g. TEACUP")
            view = rules_overall[rules_overall["lift"] >= min_lift]
            if query:
                q = query.strip().lower()
                view = view[view["antecedents"].str.lower().str.contains(q, regex=False)
                            | view["consequents"].str.lower().str.contains(q, regex=False)]
            st.caption(f"{len(view):,} rules match the current filters.")

            if not view.empty:
                top = view.head(top_n).copy()
                top["Rule"] = (top["antecedents"].str.strip() + "  →  " + top["consequents"].str.strip()).str.slice(0, 90)
                top = top.iloc[::-1]
                fig = px.bar(top, x="lift", y="Rule", orientation="h", title="Strongest rules by lift",
                             color_discrete_sequence=[NAVY], hover_data={"support": ":.3f", "confidence": ":.2f"})
                fig.update_layout(height=max(320, 34 * len(top) + 120), xaxis_title="Lift", yaxis_title=None)
                st.plotly_chart(fig, width="stretch")

                fig = px.scatter(view, x="support", y="confidence", color="lift", size="lift",
                                 color_continuous_scale="Blues", title="Support vs confidence (colour = lift)",
                                 hover_data=["antecedents", "consequents"])
                fig.update_layout(height=420)
                st.plotly_chart(fig, width="stretch")

                st.dataframe(
                    view[["antecedents", "consequents", "support", "confidence", "lift"]],
                    hide_index=True, width="stretch", height=360,
                    column_config={
                        "support": st.column_config.NumberColumn(format="%.3f"),
                        "confidence": st.column_config.NumberColumn(format="%.2f"),
                        "lift": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

    # ---- by segment ----
    with sub_seg:
        st.caption(
            f"Run separately inside each segment on its top {SEG_TOP_ITEMS} products "
            f"(min support {SEG_MIN_SUPPORT}, min lift {SEG_MIN_LIFT}). Segments with fewer than "
            f"{MIN_CUSTOMERS_FOR_RELIABLE_APRIORI} customers are skipped because Apriori on so few customers "
            "produces rules that look perfect but mean nothing."
        )
        st.dataframe(seg_status, hide_index=True, width="stretch")

        seg_choice = st.selectbox("Segment", list(summary["SegmentName"]))
        status_row = seg_status[seg_status["Segment"] == seg_choice]
        if status_row.empty or str(status_row.iloc[0]["Status"]).startswith("Skipped"):
            n = int(status_row.iloc[0]["Customers"]) if not status_row.empty else 0
            st.warning(f"**{seg_choice}** was skipped: only {n} customers, too small a sample for reliable rules.")
        elif seg_rules.empty or seg_rules[seg_rules["Segment"] == seg_choice].empty:
            st.info("No rules passed the thresholds for this segment.")
        else:
            sr = seg_rules[seg_rules["Segment"] == seg_choice]
            k1, k2, k3 = st.columns(3)
            k1.metric("Rules found", f"{len(sr):,}")
            k2.metric("Strongest lift", f"{sr['lift'].max():.2f}")
            k3.metric("Unique to this segment", f"{int((sr['Pattern'] == 'Unique to this segment').sum()):,}")
            top = sr.head(10).copy()
            top["Rule"] = (top["antecedents"].str.strip() + "  →  " + top["consequents"].str.strip()).str.slice(0, 90)
            top = top.iloc[::-1]
            fig = px.bar(top, x="lift", y="Rule", orientation="h", color="Pattern",
                         color_discrete_map={"Matches global pattern": STEEL, "Unique to this segment": "#C8A951"},
                         title=f"Strongest rules: {seg_choice}")
            fig.update_layout(height=max(320, 34 * len(top) + 140), xaxis_title="Lift", yaxis_title=None,
                              legend_title=None)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                sr[["antecedents", "consequents", "support", "confidence", "lift", "Pattern"]],
                hide_index=True, width="stretch", height=340,
                column_config={
                    "support": st.column_config.NumberColumn(format="%.3f"),
                    "confidence": st.column_config.NumberColumn(format="%.2f"),
                    "lift": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.caption("'Matches global pattern' means the same rule is in the top "
                       f"{GLOBAL_TOP_PAIRS_N} overall rules by lift.")

# ----------------------------------------------------------------------------
# Business impact
# ----------------------------------------------------------------------------
with tab_impact:
    st.subheader("Projected business impact")
    st.info(
        "These are **projections under assumed conversion rates**, not measured campaign results. "
        "The defaults below are the assumptions used in the notebook."
    )
    a1, a2 = st.columns(2)
    cross_pct = a1.slider("Cross-sell conversion (%)", 0, 100, round(CONVERSION_ASSUMPTION * 100), 5)
    win_pct = a2.slider("Win-back conversion (%)", 0, 100, round(WINBACK_CONVERSION * 100), 5)
    imp = compute_impact(R, cross_pct / 100, win_pct / 100)
    if (cross_pct, win_pct) != (round(CONVERSION_ASSUMPTION * 100), round(WINBACK_CONVERSION * 100)):
        st.caption("You changed the assumptions, so these numbers differ from the notebook's default scenario.")

    k1, k2, k3 = st.columns(3)
    k1.metric("Current total revenue", money(imp["total_revenue"]))
    k2.metric("Total projected uplift", money(imp["total_impact"]))
    k3.metric("Uplift as % of revenue", f"{imp['pct']:.2f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("**Cross-sell (Apriori-driven)**")
            st.write(f"Orders (invoices) with exactly 2 of the listed teacup products: **{imp['potential_customers']:,}**")
            st.write(f"Assumed {cross_pct}% conversion: **{imp['projected_conversions']:,}**")
            st.write(f"Average teacup line revenue: **{money(imp['avg_line_revenue'])}**")
            st.write(f"Projected additional revenue: **{money(imp['projected_revenue_crosssell'])}**")
    with col_b:
        with st.container(border=True):
            st.markdown("**Win-back (K-Means-driven)**")
            st.write(f"At-Risk customers: **{imp['at_risk_customers']:,}**")
            st.write(f"Assumed {win_pct}% conversion: **{imp['winback_customers']:,}**")
            st.write(f"Uplift per won-back customer (Regular avg − At-Risk avg spend): "
                     f"**{money(imp['revenue_uplift_per_customer'])}**")
            st.write(f"Projected additional revenue: **{money(imp['projected_winback_revenue'])}**")

    comp = pd.DataFrame({
        "Opportunity": ["Cross-sell (teacup bundle)", "Win-back (At-Risk customers)"],
        "Projected revenue": [imp["projected_revenue_crosssell"], imp["projected_winback_revenue"]],
    })
    fig = px.bar(comp, x="Opportunity", y="Projected revenue", color="Opportunity",
                 color_discrete_sequence=[SAGE, STEEL], title="Where the projected uplift comes from")
    fig.update_traces(hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>")
    fig.update_layout(showlegend=False, yaxis_tickprefix="£", xaxis_title=None, yaxis_title=None, height=360)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Before vs after")
    before_after = pd.DataFrame([
        {"Area": "Cross-selling (Apriori)",
         "Before": "Teacup sets bought together only by chance; no bundle offer",
         "After": f"{imp['potential_customers']} eligible orders targeted with 'complete your set' offer",
         "Projected Revenue Impact": f"£{imp['projected_revenue_crosssell']:,.0f}"},
        {"Area": "Customer Retention (K-Means)",
         "Before": "All customers treated identically; At-Risk customers unaddressed",
         "After": f"{imp['at_risk_customers']} At-Risk customers targeted with win-back offer",
         "Projected Revenue Impact": f"£{imp['projected_winback_revenue']:,.0f}"},
        {"Area": "TOTAL",
         "Before": "Reactive, one-size-fits-all marketing",
         "After": "Data-driven, segment + pattern-aware targeting",
         "Projected Revenue Impact": f"£{imp['total_impact']:,.0f} (+{imp['pct']:.2f}%)"},
    ])
    st.dataframe(before_after, hide_index=True, width="stretch")

    with st.expander("Teacup product check (which product names the cross-sell step actually matched)"):
        st.markdown(
            "The cross-sell step filters the data with the exact product names listed in the notebook. "
            "This table shows how many rows each name matched in your data."
        )
        chk = pd.DataFrame({"Name used in the notebook": list(R["teacup_match_counts"].keys()),
                            "Rows matched": list(R["teacup_match_counts"].values())})
        st.dataframe(chk, hide_index=True, width="stretch")
        st.markdown("**Teacup product names that exist in the data** (shown with quotes so trailing spaces are visible):")
        st.code("\n".join(R["teacup_variants"]) or "(none found)")
        unmatched = [t for t, n in R["teacup_match_counts"].items() if n == 0]
        if unmatched:
            st.warning(
                "These names matched no rows, so they are not part of the cross-sell calculation: "
                + ", ".join(unmatched)
                + ". Compare them with the names above (a trailing space is a common cause)."
            )

# ----------------------------------------------------------------------------
# Insights (all text is generated from the computed results)
# ----------------------------------------------------------------------------
with tab_insights:
    st.subheader("Business insights")
    s = summary.set_index("SegmentName")
    at = s.loc["At-Risk / Hibernating"]
    reg = s.loc["Regular Customers"]
    champ = s.loc["Champions"]
    loyal = s.loc["Loyal Customers"]
    top_rule = rules_overall.iloc[0] if not rules_overall.empty else None

    insights = [
        (
            "Revenue is concentrated in a small group of customers",
            f"Champions and Loyal customers are {champ['PctCustomers'] + loyal['PctCustomers']:.1f}% of customers "
            f"but {champ['PctRevenue'] + loyal['PctRevenue']:.1f}% of revenue.",
        ),
        (
            "A quarter of the customer base has gone quiet",
            f"{int(at['Customers']):,} At-Risk customers ({at['PctCustomers']}% of customers) have an average "
            f"recency of {at['AvgRecency']:.0f} days and contribute only {at['PctRevenue']}% of revenue.",
        ),
        (
            "The win-back gap is large",
            f"Regular customers spend {money(reg['AvgMonetary'])} on average against {money(at['AvgMonetary'])} "
            f"for At-Risk customers, a gap of {money(reg['AvgMonetary'] - at['AvgMonetary'])} per customer.",
        ),
    ]
    if top_rule is not None:
        insights.append((
            "Some products are strongly tied together",
            f"The strongest rule is IF [{top_rule['antecedents'].strip()}] THEN [{top_rule['consequents'].strip()}] "
            f"with lift {top_rule['lift']:.2f}, i.e. far more often than chance.",
        ))
    insights.append((
        "Small segments need care",
        f"{champ.name} has only {int(champ['Customers'])} customers, below the minimum of "
        f"{MIN_CUSTOMERS_FOR_RELIABLE_APRIORI} used for reliable basket analysis, so it is excluded from the "
        "segment-wise rules.",
    ))
    insights.append((
        "Projected opportunity (under assumed conversion rates)",
        f"Win-back plus the teacup bundle projects {money(impact_default['total_impact'], 0)} "
        f"({impact_default['pct']:.2f}% of current revenue) using {CONVERSION_ASSUMPTION:.0%} and "
        f"{WINBACK_CONVERSION:.0%} conversion assumptions. A real campaign test is needed to confirm.",
    ))
    for title, body in insights:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)

# ----------------------------------------------------------------------------
# Method & data
# ----------------------------------------------------------------------------
with tab_method:
    st.subheader("Parameters used (from the notebook)")
    params = pd.DataFrame([
        ("RFM snapshot date", f"{R['snapshot_date'].date()} (last transaction + 1 day)"),
        ("Scaling", "StandardScaler on Recency, Frequency, Monetary"),
        ("K-Means", f"k = {K_FINAL}, random_state = {RANDOM_STATE}, n_init = {N_INIT}"),
        ("k tested", f"{K_RANGE.start} to {K_RANGE.stop - 1} (elbow + silhouette)"),
        ("Cluster to segment map", ", ".join(f"{k} = {v}" for k, v in sorted(NAME_MAP.items()))),
        ("Overall Apriori", f"top {OVERALL_TOP_ITEMS} products, min_support = {OVERALL_MIN_SUPPORT}, lift ≥ {OVERALL_MIN_LIFT}"),
        ("Segment Apriori", f"top {SEG_TOP_ITEMS} products, min_support = {SEG_MIN_SUPPORT}, lift ≥ {SEG_MIN_LIFT}"),
        ("Minimum customers for segment rules", str(MIN_CUSTOMERS_FOR_RELIABLE_APRIORI)),
        ("Cross-sell conversion assumption", f"{CONVERSION_ASSUMPTION:.0%}"),
        ("Win-back conversion assumption", f"{WINBACK_CONVERSION:.0%}"),
    ], columns=["Parameter", "Value"])
    st.dataframe(params, hide_index=True, width="stretch")

    st.subheader("Cluster to segment check")
    if R["mapping_problems"]:
        for p in R["mapping_problems"]:
            st.warning(p)
    else:
        st.success("Segment names line up with the cluster profiles (Champions has the highest average spend, "
                   "At-Risk / Hibernating the highest average recency).")

    st.subheader("Raw data quality")
    q = pd.DataFrame({"Check": list(R["raw_quality"].keys()), "Count": list(R["raw_quality"].values())})
    st.dataframe(q, hide_index=True, width="stretch")
    st.caption(f"Raw file: {R['raw_rows']:,} rows × {R['raw_cols']} columns.")

    st.subheader("Cleaning steps")
    st.dataframe(pd.DataFrame(R["funnel"], columns=["Step", "Rows"]), hide_index=True, width="stretch")

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Customer segments (CSV)", rfm.to_csv(index=False), "customer_segments.csv", "text/csv")
    d2.download_button("Segment summary (CSV)", summary.to_csv(), "segment_summary.csv", "text/csv")
    d3.download_button("Overall rules (CSV)", rules_overall.to_csv(index=False),
                       "association_rules_overall.csv", "text/csv")
    d4, d5, d6 = st.columns(3)
    if not seg_rules.empty:
        d4.download_button("Segment rules (CSV)", seg_rules.to_csv(index=False),
                           "association_rules_by_segment.csv", "text/csv")
    d5.download_button("Before vs after (CSV)", before_after.to_csv(index=False),
                       "before_after_summary.csv", "text/csv")
    d6.download_button("Monthly revenue (CSV)", R["monthly"].to_csv(), "monthly_revenue.csv", "text/csv")

    st.subheader("Environment")
    st.caption(" · ".join(f"{p} {pkg_version(p)}" for p in
                          ["pandas", "scikit-learn", "mlxtend", "streamlit", "plotly"]))
