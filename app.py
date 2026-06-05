import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import where
import geopandas
import geoplot
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Electricity Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Global Electricity Generation & Consumption")
st.markdown(
    "Analysis of electricity production, consumption, renewables, and CO₂ emissions "
    "across 48 OECD countries · Data: IEA Monthly Electricity Statistics (2024)"
)
st.divider()

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_monthly_elec = pd.read_csv("MES_0124.csv", skiprows=8)
    df_world_pop = pd.read_csv("WorldPopulation2023.csv")
    df_carbon_emissions = pd.read_csv("co2_emissions_kt_by_country.csv")

    # --- Clean electricity data ---
    df_monthly_elec["Time"] = pd.to_datetime(df_monthly_elec["Time"])
    df_monthly_elec["Year"] = df_monthly_elec["Time"].dt.year
    df_monthly_elec = df_monthly_elec[~df_monthly_elec["Country"].str.contains("OECD")]
    df_monthly_elec = df_monthly_elec[~df_monthly_elec["Country"].str.contains("IEA")]

    # --- Clean population data ---
    df_world_pop = df_world_pop[["Country", "Population2023"]]
    df_world_pop["Country_updated"] = df_world_pop["Country"]
    replacements = {
        "China": "People's Republic of China",
        "South Korea": "Korea",
        "Turkey": "Republic of Turkiye",
        "Czech Republic (Czechia)": "Czech Republic",
        "Slovakia": "Slovak Republic",
    }
    for old, new in replacements.items():
        df_world_pop["Country_updated"] = where(
            df_world_pop["Country"] == old, new, df_world_pop["Country_updated"]
        )

    return df_monthly_elec, df_world_pop, df_carbon_emissions


with st.spinner("Loading datasets…"):
    try:
        df_monthly_elec, df_world_pop, df_carbon_emissions = load_data()
        data_loaded = True
    except FileNotFoundError as e:
        data_loaded = False
        missing_file = str(e)

if not data_loaded:
    st.error(
        f"**Data file not found:** `{missing_file}`\n\n"
        "Please place the following CSV files in the **same folder** as `app.py`:\n"
        "- `MES_0124.csv` — IEA Monthly Electricity Statistics\n"
        "- `WorldPopulation2023.csv` — World Population 2023 (Kaggle)\n"
        "- `co2_emissions_kt_by_country.csv` — CO₂ Emissions by Country (Kaggle)"
    )
    st.stop()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Q1 · Top 10 Final Consumption (2023)",
        "Q2 · Top 10 Net Production (2023)",
        "Q3 · Energy Mix 2015–2023",
        "Q4 · Renewables Share Over Time",
        "Q5 · Production vs Population",
        "Q6 · Per-Capita Production (2023)",
        "Q7 · Global Seasonality",
        "Q8 · Australia Seasonality",
        "Q9 · CO₂ Emissions World Map",
        "Q10 · Combustible Fuels vs CO₂",
    ],
)
sns.set()

# ── Q1 ────────────────────────────────────────────────────────────────────────
if section == "Q1 · Top 10 Final Consumption (2023)":
    st.header("Q1 · Top 10 Countries by Final Electricity Consumption (2023)")
    st.markdown(
        "Which countries in the OECD dataset had the highest **Final Consumption** "
        "of electricity in 2023? *(Note: China and India are excluded — only Net "
        "Production data is available for them.)*"
    )

    df_2023 = df_monthly_elec[
        (df_monthly_elec["Time"] >= "2023-01-01")
        & (df_monthly_elec["Time"] <= "2023-12-31")
    ]
    df_2023_consumption = df_2023[
        df_2023["Balance"] == "Final Consumption (Calculated)"
    ]
    df_2023_consumption = (
        df_2023_consumption.groupby(["Country", "Balance", "Product", "Year", "Unit"])
        .sum(numeric_only=True)
        .fillna(0)
        .sort_values(by="Value", ascending=False)
        .head(10)
    )

    df_plot = df_2023_consumption.reset_index().sort_values("Value")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_plot["Country"], df_plot["Value"], color="steelblue")
    ax.set_title("Final Electricity Consumption – Top 10 OECD Nations (2023)")
    ax.set_xlabel("Final Consumption (GWh)")
    ax.set_ylabel("Country")
    st.pyplot(fig)
    st.dataframe(df_plot[["Country", "Value"]].sort_values("Value", ascending=False).reset_index(drop=True))

# ── Q2 ────────────────────────────────────────────────────────────────────────
elif section == "Q2 · Top 10 Net Production (2023)":
    st.header("Q2 · Top 10 Countries by Net Electricity Production (2023)")

    df_2023 = df_monthly_elec[
        (df_monthly_elec["Time"] >= "2023-01-01")
        & (df_monthly_elec["Time"] <= "2023-12-31")
    ]
    df_2023_production = df_2023[df_2023["Balance"] == "Net Electricity Production"]
    df_2023_production = (
        df_2023_production.groupby(["Country", "Year", "Unit", "Balance", "Product"])
        .sum(numeric_only=True)
        .unstack()
        .fillna(0)
    )
    df_2023_production.columns = df_2023_production.columns.droplevel()

    df_2023_production["Total_Products_Production"] = (
        df_2023_production["Total Renewables (Hydro, Geo, Solar, Wind, Other)"]
        + df_2023_production["Coal, Peat and Manufactured Gases"]
        + df_2023_production["Natural Gas"]
        + df_2023_production["Oil and Petroleum Products"]
        + df_2023_production["Nuclear"]
        + df_2023_production["Other Combustible Non-Renewables"]
        + df_2023_production["Not Specified"]
    )
    df_top10 = (
        df_2023_production[["Total_Products_Production"]]
        .sort_values("Total_Products_Production", ascending=False)
        .head(10)
    )
    df_plot = df_top10.reset_index().sort_values("Total_Products_Production")
    # flatten multi-index if needed
    if df_plot.columns.nlevels > 1:
        df_plot.columns = ["_".join(filter(None, c)) if isinstance(c, tuple) else c for c in df_plot.columns]

    country_col = [c for c in df_plot.columns if "Country" in c][0]
    val_col = [c for c in df_plot.columns if "Total_Products" in c or "Production" in c][0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_plot[country_col], df_plot[val_col], color="steelblue")
    ax.set_title("Total Net Electricity Production – Top 10 (2023)")
    ax.set_xlabel("Net Production (GWh)")
    st.pyplot(fig)
    st.dataframe(df_plot[[country_col, val_col]].sort_values(val_col, ascending=False).reset_index(drop=True))

# ── Q3 ────────────────────────────────────────────────────────────────────────
elif section == "Q3 · Energy Mix 2015–2023":
    st.header("Q3 · Energy Mix in Electricity Production (2015–2023)")
    st.markdown("How has the usage of Coal, Gas, Solar, Wind, Nuclear, and others changed over time?")

    df_decade = df_monthly_elec[
        (df_monthly_elec["Time"] >= "2015-01-01")
        & (df_monthly_elec["Time"] <= "2023-12-31")
    ]
    df_decade_production = df_decade[df_decade["Balance"] == "Net Electricity Production"]
    df_decade_production = df_decade_production[
        df_decade_production["Country"] != "Costa Rica"
    ]

    df_agg = (
        df_decade_production.groupby(["Year", "Unit", "Balance", "Product"])
        .sum(numeric_only=True)
        .unstack()
        .fillna(0)
    )
    df_agg.columns = df_agg.columns.droplevel()
    df_agg["Others"] = (
        df_agg["Electricity"]
        - df_agg["Coal, Peat and Manufactured Gases"]
        - df_agg["Natural Gas"]
        - df_agg["Solar"]
        - df_agg["Wind"]
        - df_agg["Nuclear"]
    )
    df_sel = df_agg[
        ["Coal, Peat and Manufactured Gases", "Natural Gas", "Solar", "Wind", "Nuclear", "Others"]
    ]
    df_sel.index = df_sel.index.droplevel([1, 2])

    color_map = {
        "Coal, Peat and Manufactured Gases": "red",
        "Natural Gas": "orange",
        "Solar": "gold",
        "Wind": "green",
        "Nuclear": "steelblue",
        "Others": "grey",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sel.plot(kind="area", ax=ax, color=[color_map[c] for c in df_sel.columns])
    ax.set_title("Types of Energy Generation (2015–2023)")
    ax.set_ylabel("Total GWh")
    ax.set_xlabel("Year")
    st.pyplot(fig)

# ── Q4 ────────────────────────────────────────────────────────────────────────
elif section == "Q4 · Renewables Share Over Time":
    st.header("Q4 · Share of Renewables in Electricity Production (2015–2023)")

    all_countries = sorted(df_monthly_elec["Country"].unique())
    selected_country = st.selectbox("Select a country", all_countries, index=list(all_countries).index("Australia") if "Australia" in all_countries else 0)

    df_decade = df_monthly_elec[
        (df_monthly_elec["Time"] >= "2015-01-01")
        & (df_monthly_elec["Time"] <= "2023-12-31")
    ]
    df_decade_production = df_decade[df_decade["Balance"] == "Net Electricity Production"]

    def compute_renewables_share(df_prod):
        df = (
            df_prod.groupby(["Year", "Unit", "Balance", "Product"])
            .sum(numeric_only=True)
            .unstack()
            .fillna(0)
        )
        df.columns = df.columns.droplevel()
        df["Total Non-renewables"] = (
            df["Electricity"] - df["Total Renewables (Hydro, Geo, Solar, Wind, Other)"]
        )
        df["Renewables Share"] = (
            df["Total Renewables (Hydro, Geo, Solar, Wind, Other)"] / df["Electricity"]
        )
        df["Non-renewables Share"] = df["Total Non-renewables"] / df["Electricity"]
        df.index = df.index.droplevel([1, 2])
        return df[["Renewables Share", "Non-renewables Share"]]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{selected_country}")
        df_country = df_decade_production[df_decade_production["Country"] == selected_country]
        if df_country.empty:
            st.warning("No data available for this country.")
        else:
            df_aus = compute_renewables_share(df_country)
            fig, ax = plt.subplots(figsize=(6, 4))
            df_aus.plot(kind="area", ax=ax, ylabel="Share", title=f"{selected_country} – Renewables vs Non-renewables")
            st.pyplot(fig)

    with col2:
        st.subheader("All 48 Countries (aggregate)")
        df_all = compute_renewables_share(df_decade_production)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        df_all.plot(kind="area", ax=ax2, ylabel="Share", title="All Countries – Renewables vs Non-renewables")
        st.pyplot(fig2)

# ── Q5 ────────────────────────────────────────────────────────────────────────
elif section == "Q5 · Production vs Population":
    st.header("Q5 · Electricity Production vs Country Population (2023)")

    df_2023 = df_monthly_elec[
        (df_monthly_elec["Time"] >= "2023-01-01")
        & (df_monthly_elec["Time"] <= "2023-12-31")
    ]
    df_2023_prod = df_2023[
        (df_2023["Balance"] == "Net Electricity Production")
        & (df_2023["Product"] == "Electricity")
    ]
    df_2023_prod = (
        df_2023_prod.groupby(["Country", "Year", "Unit", "Balance"])
        .sum(numeric_only=True)
        .reset_index()
    )

    df_merged = df_world_pop.merge(df_2023_prod, how="right", left_on="Country_updated", right_on="Country")
    df_merged = df_merged.dropna(subset=["Population2023"])

    exclude = st.multiselect(
        "Exclude outlier countries",
        options=sorted(df_merged["Country"].unique()),
        default=["People's Republic of China", "United States", "India"],
    )
    df_plot = df_merged[~df_merged["Country"].isin(exclude)]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=df_plot, x="Population2023", y="Value", ax=ax, scatter_kws={"alpha": 0.7})
    ax.set_title("Population vs Net Electricity Production (2023)")
    ax.set_xlabel("Population 2023")
    ax.set_ylabel("Net Electricity Production (GWh)")
    for _, row in df_plot.iterrows():
        ax.annotate(row["Country"], (row["Population2023"], row["Value"]), fontsize=6, alpha=0.7)
    st.pyplot(fig)

# ── Q6 ────────────────────────────────────────────────────────────────────────
elif section == "Q6 · Per-Capita Production (2023)":
    st.header("Q6 · Electricity Production Per Capita (2023)")
    st.markdown("Which country produces the most electricity per person?")

    df_2023 = df_monthly_elec[
        (df_monthly_elec["Time"] >= "2023-01-01")
        & (df_monthly_elec["Time"] <= "2023-12-31")
    ]
    df_2023_prod = df_2023[
        (df_2023["Balance"] == "Net Electricity Production")
        & (df_2023["Product"] == "Electricity")
    ]
    df_2023_prod = (
        df_2023_prod.groupby(["Country"])
        .sum(numeric_only=True)
        .reset_index()[["Country", "Value"]]
    )

    df_merged = df_world_pop.merge(df_2023_prod, left_on="Country_updated", right_on="Country")
    df_merged["per_person_MWh"] = df_merged["Value"] / df_merged["Population2023"] * 1000

    top_n = st.slider("Show top N countries", 5, 20, 10)
    df_top = df_merged.sort_values("per_person_MWh", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df_top.sort_values("per_person_MWh")["Country_updated"],
            df_top.sort_values("per_person_MWh")["per_person_MWh"], color="teal")
    ax.set_title(f"Electricity Production Per Capita – Top {top_n} (2023)")
    ax.set_xlabel("MWh per person")
    st.pyplot(fig)
    st.dataframe(df_top[["Country_updated", "per_person_MWh"]].rename(columns={"Country_updated": "Country", "per_person_MWh": "MWh per person"}).reset_index(drop=True))

# ── Q7 ────────────────────────────────────────────────────────────────────────
elif section == "Q7 · Global Seasonality":
    st.header("Q7 · Global Yearly Seasonality of Electricity Production (2015–2023)")

    df_decade = df_monthly_elec[
        (df_monthly_elec["Time"] >= "2015-01-01")
        & (df_monthly_elec["Time"] <= "2023-12-31")
    ]
    df_prod = df_decade[
        (df_decade["Balance"] == "Net Electricity Production")
        & (df_decade["Product"] == "Electricity")
    ].copy()
    df_prod["Month"] = df_prod["Time"].dt.month

    df_seasonal = (
        df_prod.groupby(["Month", "Balance", "Year"])
        .sum(numeric_only=True)
        .unstack()
    )
    df_seasonal.index = df_seasonal.index.droplevel(1)

    fig, ax = plt.subplots(figsize=(12, 6))
    df_seasonal.plot(kind="line", ax=ax)
    ax.set_title("Yearly Seasonality of Total Electricity Production (all countries)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total GWh")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    st.pyplot(fig)

# ── Q8 ────────────────────────────────────────────────────────────────────────
elif section == "Q8 · Australia Seasonality":
    st.header("Q8 · Yearly Seasonality – Australia (2015–2023)")

    all_countries = sorted(df_monthly_elec["Country"].unique())
    selected = st.selectbox("Select a country to compare", all_countries,
                            index=list(all_countries).index("Australia") if "Australia" in all_countries else 0)

    df_decade = df_monthly_elec[
        (df_monthly_elec["Time"] >= "2015-01-01")
        & (df_monthly_elec["Time"] <= "2023-12-31")
    ]
    df_prod = df_decade[
        (df_decade["Balance"] == "Net Electricity Production")
        & (df_decade["Product"] == "Electricity")
    ].copy()
    df_prod["Month"] = df_prod["Time"].dt.month

    df_country = df_prod[df_prod["Country"] == selected].copy()
    df_seasonal = (
        df_country.groupby(["Month", "Balance", "Year"])
        .sum(numeric_only=True)
        .unstack()
    )
    df_seasonal.index = df_seasonal.index.droplevel(1)

    fig, ax = plt.subplots(figsize=(12, 6))
    df_seasonal.plot(kind="line", ax=ax)
    ax.set_title(f"Yearly Seasonality – {selected}")
    ax.set_xlabel("Month")
    ax.set_ylabel("GWh")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    st.pyplot(fig)

# ── Q9 ────────────────────────────────────────────────────────────────────────
elif section == "Q9 · CO₂ Emissions World Map":
    st.header("Q9 · CO₂ Emissions Per Capita by Country (2019)")

    df_co2_2019 = df_carbon_emissions[df_carbon_emissions["year"] == 2019]

    try:
        world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
        world = world[world.name != "Antarctica"]
        world.rename({"iso_a3": "country_code"}, axis=1, inplace=True)
        world["country_code"] = where(world["name"] == "France", "FRA", world["country_code"])
        world["country_code"] = where(world["name"] == "Norway", "NOR", world["country_code"])
        world = world.merge(df_co2_2019, on=["country_code"], how="left")
        world["emissions_per_capita"] = world["value"] / world["pop_est"]

        fig, ax = plt.subplots(1, figsize=(14, 7))
        ax.set_title("CO₂ Emissions per Capita by Country (2019)", fontsize=16)
        ax.axis("off")
        world.plot(
            column="emissions_per_capita",
            legend=True,
            ax=ax,
            cmap="plasma",
            missing_kwds={"color": "lightgrey"},
            legend_kwds={"label": "CO₂ per Capita (kt)", "orientation": "horizontal"},
        )
        st.pyplot(fig)

        st.subheader("Top 10 countries by CO₂ per capita (2019)")
        top10 = (
            world[["country_name", "emissions_per_capita"]]
            .sort_values("emissions_per_capita", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        st.dataframe(top10)
    except Exception as e:
        st.error(f"Could not render map: {e}\n\nThis may be due to a missing geopandas/geoplot dependency.")

# ── Q10 ───────────────────────────────────────────────────────────────────────
elif section == "Q10 · Combustible Fuels vs CO₂":
    st.header("Q10 · Combustible Fuel Electricity vs CO₂ Emissions (2019)")
    st.markdown("Is there a relationship between how much electricity a country generates from combustible fuels and its total CO₂ emissions?")

    df_2019 = df_monthly_elec[
        (df_monthly_elec["Time"] >= "2019-01-01")
        & (df_monthly_elec["Time"] <= "2019-12-31")
    ]
    df_2019_prod = df_2019[df_2019["Product"] == "Total Combustible Fuels"]
    df_2019_prod = (
        df_2019_prod.groupby(["Country", "Year", "Unit", "Balance", "Product"])
        .sum(numeric_only=True)
        .reset_index()
    )
    for lvl in [4, 3, 2, 1]:
        if df_2019_prod.index.nlevels > 1:
            df_2019_prod.index = df_2019_prod.index.droplevel(lvl)

    df_co2_2019 = df_carbon_emissions[df_carbon_emissions["year"] == 2019]
    try:
        world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
        world.rename({"iso_a3": "country_code"}, axis=1, inplace=True)
        world["country_code"] = where(world["name"] == "France", "FRA", world["country_code"])
        world["country_code"] = where(world["name"] == "Norway", "NOR", world["country_code"])
        world = world.merge(df_co2_2019, on=["country_code"], how="left")

        df_merged = df_2019_prod.merge(world, how="left", left_on="Country", right_on="country_name")
        df_merged = df_merged.dropna(subset=["value"])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(data=df_merged, x="Value", y="value", ax=ax, scatter_kws={"alpha": 0.7})
        ax.set_title("Combustible Fuel Electricity Production vs CO₂ Emissions (2019)")
        ax.set_xlabel("Electricity from Combustible Fuels (GWh)")
        ax.set_ylabel("CO₂ Emissions (kt)")
        for _, row in df_merged.iterrows():
            ax.annotate(row["Country"], (row["Value"], row["value"]), fontsize=6, alpha=0.6)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not load geographic data for CO₂ merge: {e}")
