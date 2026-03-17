“”“Quality assurance visualisation for input data”””

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import slugify

import assets.data.house_prices.scottish_prices
from common.owners import AssetOwners
from system.assets import SecurityLevel, VisualAsset
from system.output import matplotlib_output_generator

NAME = “Scottish prices - transactions by property type and year”

DESCRIPTION = (
“Heatmap of transaction counts broken down by property type and year for Scotland. “
“Used to QA coverage and spot missing data or unexpected gaps.”
)

HOUSE_TYPE_LABELS = {
“D”: “Detached”,
“S”: “Semi-detached”,
“T”: “Terraced”,
“F”: “Flat”,
“O”: “Other”,
}

def data_pipeline(data) -> pd.DataFrame:
“”“Process Scottish price data into property type x year transaction counts.”””

```
data = data["scottish_prices"][0]["file_object"]
df = pd.read_csv(data, usecols=["Entry_Date", "House_Type"])
df = df.rename(columns={"Entry_Date": "entry_date", "House_Type": "house_type"})

df["year"] = pd.to_datetime(df["entry_date"], format="%Y-%m-%d").dt.year.astype(str)
df["house_type"] = df["house_type"].map(HOUSE_TYPE_LABELS).fillna(df["house_type"])

df = (
    df.groupby(["year", "house_type"])
    .agg(transaction_count=("house_type", "count"))
    .reset_index()
)

# Pivot to matrix form for heatmap
matrix = df.pivot(index="house_type", columns="year", values="transaction_count")

return matrix
```

def create_chart(matrix: pd.DataFrame):
fig, ax = plt.subplots(figsize=(12, 5))

```
sns.heatmap(
    matrix,
    annot=True,
    fmt=",",
    cmap="Blues",
    linewidths=0.5,
    linecolor="white",
    ax=ax,
    cbar_kws={"label": "Transaction Count"},
)

ax.set_title(
    "Transaction count by property type and year",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Property Type", fontsize=11)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
return fig
```

def chart_pipeline(data):
slug = slugify.slugify(NAME)
matrix = data_pipeline(data)
fig = create_chart(matrix)
yield from matplotlib_output_generator(fig, f”{slug}-dynamic”, “Dynamic”)

asset = VisualAsset(
name=NAME,
description=DESCRIPTION,
security_level=SecurityLevel.SENSITIVE,
upstream_data_assets={
“scottish_prices”: assets.data.house_prices.scottish_prices.asset
},
data_pipeline=data_pipeline,
chart_pipeline=chart_pipeline,
owners=[AssetOwners.POLARIS_TEAM],
)

if **name** == “**main**”:
from system.input import generate_chart_local

```
generate_chart_local(asset)
```
