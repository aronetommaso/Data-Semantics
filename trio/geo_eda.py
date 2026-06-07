"""
geo_eda.py
Geographic EDA — ACLED Middle East Conflict Dataset (2015-2023)

Produces 7 publication-quality plots:
  1.  Choropleth — total events per country
  2.  Choropleth — total fatalities per country (log scale)
  3.  Hex-bin density map — all conflict events (lat/lon)
  4.  Animated scatter map — events per year (Plotly HTML)
  5.  Fatality hotspot bubble map (Plotly HTML + PNG)
  6.  Event-type stacked bar per country
  7.  Temporal heatmap — events per country per year

Setup (one-time):
    pip install geopandas plotly kaleido matplotlib seaborn folium

Run:
    python geo_eda.py
"""

import os
import urllib.request
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH    = r"C:\Users\HP\Desktop\data_science\primo_anno\DataSemantics\Project\Data-Semantics\acled_unified_middle_east.csv"
OUTPUT_DIR   = r"C:\Users\HP\Desktop\data_science\primo_anno\DataSemantics\Project\Data-Semantics\trio\geo_output"
GEOJSON_PATH = r"C:\Users\HP\Desktop\data_science\primo_anno\DataSemantics\Project\Data-Semantics\trio\countries.geojson"
GEOJSON_URL  = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
    "/master/geojson/ne_110m_admin_0_countries.geojson"
)
PALETTE      = "YlOrRd"
ACCENT       = "#16324F"
TEAL         = "#2A7F7F"
DPI          = 180

ISO3_MAP = {
    "Palestine":            "PSE",
    "Syria":                "SYR",
    "Yemen":                "YEM",
    "Iraq":                 "IRQ",
    "Turkey":               "TUR",
    "Iran":                 "IRN",
    "Lebanon":              "LBN",
    "Israel":               "ISR",
    "Jordan":               "JOR",
    "Bahrain":              "BHR",
    "Saudi Arabia":         "SAU",
    "Qatar":                "QAT",
    "Kuwait":               "KWT",
    "United Arab Emirates": "ARE",
    "Oman":                 "OMN",
}

EVENT_COLORS = {
    "Battles":                     "#C0392B",
    "Explosions/Remote violence":  "#E67E22",
    "Violence against civilians":  "#8E44AD",
    "Riots":                       "#2980B9",
    "Protests":                    "#27AE60",
    "Strategic developments":      "#7F8C8D",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def download_geojson(path, url):
    if not os.path.exists(path):
        print(f"  Downloading world geometries from Natural Earth...")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved to {path}")

def save_msg(fname):
    # FIX: Sostituito il carattere speciale freccia con "->" per evitare errori cp1252 su Windows
    print(f"  -> saved {fname}")


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading ACLED data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df["event_date"] = pd.to_datetime(df["event_date"])
df["year"]       = df["event_date"].dt.year
df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
df["iso3"]       = df["country"].map(ISO3_MAP)
print(f"  {len(df):,} events loaded.")

# ── WORLD GEOMETRIES ──────────────────────────────────────────────────────────
download_geojson(GEOJSON_PATH, GEOJSON_URL)
world    = gpd.read_file(GEOJSON_PATH)
me_iso3  = list(ISO3_MAP.values())
me_world = world[world["ISO_A3"].isin(me_iso3)].copy()

# Fix Palestine — Natural Earth uses "PSX" or missing; remap
me_world.loc[me_world["ISO_A3"] == "PSX", "ISO_A3"] = "PSE"
# Also check via name
pal_mask = me_world["NAME"].str.contains("Palest", na=False)
me_world.loc[pal_mask, "ISO_A3"] = "PSE"


# =============================================================================
# PLOT 1 — Choropleth: Total Events per Country
# =============================================================================
print("\nPlot 1: Choropleth - events per country...")

events_pc = df.groupby("iso3").size().reset_index(name="events")
w1 = me_world.merge(events_pc, left_on="ISO_A3", right_on="iso3", how="left")
w1["events"] = w1["events"].fillna(0)

fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
w1.plot(
    column="events", ax=ax, cmap=PALETTE,
    legend=True,
    legend_kwds={"label": "Number of Events", "orientation": "horizontal",
                 "shrink": 0.55, "pad": 0.02, "format": "%,.0f"},
    missing_kwds={"color": "#E8E8E8", "label": "No data"},
    edgecolor="#555555", linewidth=0.5,
)
for _, row in w1.iterrows():
    if row["events"] > 0:
        c = row.geometry.centroid
        ax.annotate(
            f"{row['NAME']}\n{int(row['events']):,}",
            xy=(c.x, c.y), ha="center", va="center",
            fontsize=7.5, color=ACCENT, fontweight="bold",
        )
ax.set_title("Total Conflict Events by Country - Middle East (2015-2023)",
             fontsize=14, fontweight="bold", color=ACCENT, pad=14)
ax.set_axis_off()
plt.tight_layout()
out = f"{OUTPUT_DIR}/01_choropleth_events.png"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
save_msg(out)


# =============================================================================
# PLOT 2 — Choropleth: Total Fatalities (log scale)
# =============================================================================
print("Plot 2: Choropleth - fatalities per country...")

fat_pc = df.groupby("iso3")["fatalities"].sum().reset_index()
w2 = me_world.merge(fat_pc, left_on="ISO_A3", right_on="iso3", how="left")
w2["fatalities"]   = w2["fatalities"].fillna(0)
w2["_log_fat"]     = np.log1p(w2["fatalities"])

fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
w2.plot(
    column="_log_fat", ax=ax, cmap=PALETTE,
    legend=True,
    legend_kwds={"label": "Total Fatalities (log1p scale)",
                 "orientation": "horizontal", "shrink": 0.55, "pad": 0.02},
    missing_kwds={"color": "#E8E8E8"},
    edgecolor="#555555", linewidth=0.5,
)
for _, row in w2.iterrows():
    if row["fatalities"] > 0:
        c = row.geometry.centroid
        ax.annotate(
            f"{row['NAME']}\n{int(row['fatalities']):,}",
            xy=(c.x, c.y), ha="center", va="center",
            fontsize=7.5, color=ACCENT, fontweight="bold",
        )
ax.set_title(
    "Total Conflict Fatalities by Country - Middle East (2015-2023)\n"
    "(colour intensity = log scale)",
    fontsize=13, fontweight="bold", color=ACCENT, pad=14)
ax.set_axis_off()
plt.tight_layout()
out = f"{OUTPUT_DIR}/02_choropleth_fatalities.png"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
save_msg(out)


# =============================================================================
# PLOT 3 — Hex-bin Density Map
# =============================================================================
print("Plot 3: Hex-bin density map...")

fig, ax = plt.subplots(figsize=(14, 9), facecolor="#F0F4F8")
me_world.plot(ax=ax, color="#DDEEFF", edgecolor="#7799BB", linewidth=0.6, zorder=1)

hb = ax.hexbin(
    df["longitude"], df["latitude"],
    gridsize=75, cmap="inferno",
    mincnt=1, bins="log",
    alpha=0.88, linewidths=0.0, zorder=2,
)
cb = fig.colorbar(hb, ax=ax, orientation="vertical", shrink=0.65, pad=0.02)
cb.set_label("Events in cell (log scale)", fontsize=9, color=ACCENT)

xmin, ymin, xmax, ymax = me_world.total_bounds
ax.set_xlim(xmin - 1.5, xmax + 1.5)
ax.set_ylim(ymin - 1.5, ymax + 1.5)
ax.set_title("Spatial Density of Conflict Events - Middle East (2015-2023)",
             fontsize=14, fontweight="bold", color=ACCENT, pad=14)
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude", fontsize=9)
plt.tight_layout()
out = f"{OUTPUT_DIR}/03_hexbin_density.png"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
save_msg(out)


# =============================================================================
# PLOT 4 — Animated Scatter Map (Plotly HTML)
# =============================================================================
print("Plot 4: Animated scatter map (Plotly)...")

df_s = df.sample(n=min(25000, len(df)), random_state=42).copy()
df_s["dot_size"] = df_s["fatalities"].clip(upper=60) + 3

fig4 = px.scatter_geo(
    df_s.sort_values("year"),
    lat="latitude", lon="longitude",
    color="event_type",
    size="dot_size", size_max=20,
    animation_frame="year",
    hover_name="location",
    hover_data={"country": True, "event_type": True, "fatalities": True,
                "dot_size": False, "latitude": False, "longitude": False},
    color_discrete_map=EVENT_COLORS,
    projection="natural earth",
    title="Conflict Events by Type and Year - Middle East (2015-2023)",
    template="plotly_white",
)
fig4.update_geos(
    showcountries=True, countrycolor="#AAAAAA",
    showland=True, landcolor="#F5F5F5",
    showocean=True, oceancolor="#D0E8F5",
    lataxis_range=[10, 43], lonaxis_range=[26, 63],
)
fig4.update_layout(height=620, title_font_size=14, title_font_color=ACCENT,
                   legend_title_text="Event Type")
out = f"{OUTPUT_DIR}/04_animated_scatter.html"
fig4.write_html(out)
save_msg(out)


# =============================================================================
# PLOT 5 — Fatality Hotspot Bubble Map
# =============================================================================
print("Plot 5: Fatality hotspot map...")

df_lethal = df[df["fatalities"] > 0].copy()
df_lethal["lat_r"] = df_lethal["latitude"].round(1)
df_lethal["lon_r"] = df_lethal["longitude"].round(1)
agg = df_lethal.groupby(["lat_r", "lon_r", "country"]).agg(
    total_fatalities=("fatalities", "sum"),
    n_events=("event_id_cnty", "count")
).reset_index()

fig5 = px.scatter_geo(
    agg, lat="lat_r", lon="lon_r",
    size="total_fatalities", color="total_fatalities",
    color_continuous_scale="YlOrRd",
    size_max=55,
    hover_name="country",
    hover_data={"total_fatalities": True, "n_events": True,
                "lat_r": False, "lon_r": False},
    projection="natural earth",
    title="Fatality Hotspots - Middle East (2015-2023)<br>"
          "<sup>Bubble size & colour = total fatalities at location (0.1 degree grid)</sup>",
    template="plotly_white",
)
fig5.update_geos(
    showcountries=True, countrycolor="#888888",
    showland=True, landcolor="#F0F0F0",
    showocean=True, oceancolor="#D0E8F5",
    lataxis_range=[10, 43], lonaxis_range=[26, 63],
)
fig5.update_layout(height=620, title_font_size=13, title_font_color=ACCENT,
                   coloraxis_colorbar_title="Fatalities")

out_html = f"{OUTPUT_DIR}/05_fatality_hotspots.html"
fig5.write_html(out_html)
save_msg(out_html)
try:
    out_png = f"{OUTPUT_DIR}/05_fatality_hotspots.png"
    fig5.write_image(out_png, width=1400, height=700, scale=2)
    save_msg(out_png)
except Exception:
    print("  (PNG skipped - install kaleido for static export: pip install kaleido)")


# =============================================================================
# PLOT 6 — Event Type Stacked Bar per Country
# =============================================================================
print("Plot 6: Stacked bar - event types per country...")

pivot = df.groupby(["country", "event_type"]).size().unstack(fill_value=0)
pivot["_total"] = pivot.sum(axis=1)
pivot = pivot.sort_values("_total", ascending=True).drop(columns="_total")
col_order = [c for c in df["event_type"].value_counts().index if c in pivot.columns]
pivot = pivot[col_order]

fig, ax = plt.subplots(figsize=(13, 8), facecolor="white")
bottom = np.zeros(len(pivot))
for etype in pivot.columns:
    vals = pivot[etype].values
    color = EVENT_COLORS.get(etype, "#AAAAAA")
    ax.barh(pivot.index, vals, left=bottom, label=etype,
            color=color, edgecolor="white", linewidth=0.5)
    bottom += vals

# Total labels
for i, total in enumerate(pivot.sum(axis=1)):
    ax.text(total + 150, i, f"{int(total):,}", va="center",
            fontsize=8.5, color=ACCENT, fontweight="bold")

ax.set_xlabel("Number of Events", fontsize=10)
ax.set_title(
    "Conflict Event Types by Country - Middle East (2015-2023)\n"
    "(sorted by total events)",
    fontsize=13, fontweight="bold", color=ACCENT, pad=14)
ax.legend(loc="lower right", fontsize=9, framealpha=0.85,
          title="Event Type", title_fontsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=9.5)
plt.tight_layout()
out = f"{OUTPUT_DIR}/06_event_type_by_country.png"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
save_msg(out)


# =============================================================================
# PLOT 7 — Temporal Heatmap: Events per Country per Year
# =============================================================================
print("Plot 7: Temporal heatmap...")

hm = df.groupby(["country", "year"]).size().unstack(fill_value=0)
hm = hm.loc[hm.sum(axis=1).sort_values(ascending=False).index]
log_hm = np.log1p(hm.values)

fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
im = ax.imshow(log_hm, aspect="auto", cmap="YlOrRd", interpolation="nearest")

ax.set_xticks(range(len(hm.columns)))
ax.set_xticklabels(hm.columns.astype(int), fontsize=10)
ax.set_yticks(range(len(hm.index)))
ax.set_yticklabels(hm.index, fontsize=10)

for i in range(len(hm.index)):
    for j in range(len(hm.columns)):
        val = hm.values[i, j]
        if val > 0:
            txt_col = "white" if log_hm[i, j] > log_hm.max() * 0.58 else ACCENT
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=7, color=txt_col, fontweight="bold")

cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("Events (log1p scale)", fontsize=9)
cb.set_ticks([])

ax.set_title(
    "Conflict Events per Country per Year - Middle East (2015-2023)\n"
    "(colour = log scale; numbers = raw counts; sorted by total)",
    fontsize=13, fontweight="bold", color=ACCENT, pad=14)
ax.set_xlabel("Year", fontsize=10)
plt.tight_layout()
out = f"{OUTPUT_DIR}/07_temporal_heatmap.png"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
save_msg(out)


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 58)
print(f"All outputs saved in: {OUTPUT_DIR}/")
print("=" * 58)
files = [
    ("01_choropleth_events.png",     "events per country (choropleth)"),
    ("02_choropleth_fatalities.png", "fatalities per country, log scale"),
    ("03_hexbin_density.png",        "spatial event density (hex-bin)"),
    ("04_animated_scatter.html",     "events animated by year (interactive)"),
    ("05_fatality_hotspots.html",    "fatality bubbles (interactive)"),
    ("05_fatality_hotspots.png",     "fatality bubbles (static, if kaleido installed)"),
    ("06_event_type_by_country.png", "stacked bar - event types"),
    ("07_temporal_heatmap.png",      "country x year event heatmap"),
]
for fname, desc in files:
    # FIX: Sostituito spunta e trattino lungo con testo ASCII per evitare crash
    exists = "[OK]" if os.path.exists(f"{OUTPUT_DIR}/{fname}") else "[MISSING]"
    print(f"  {exists}  {fname:<40} {desc}")
print("=" * 58)