
“”“Quality assurance visualisation for input data”””

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import slugify

import assets.data.house_prices.scottish_prices
from common.owners import AssetOwners
from system.assets import SecurityLevel, VisualAsset
from system.output import matplotlib_output_generator

NAME = “Scottish prices - transaction counts by local authority”

DESCRIPTION = (
“Bar chart of transaction counts by local authority for Scotland. “
“Used to QA geographic coverage and spot missing or underrepresented areas.”
)

def data_pipeline(data) -> pd.DataFrame:
“”“Process Scottish price data into transaction counts by local authority.”””

```
data = data["scottish_prices"][0]["file_object"]
df = pd.read_csv(data, usecols=["Local_Authority"])
df = df.rename(columns={"Local_Authority": "local_authority"})

df = (
    df.groupby("local_authority")
    .agg(transaction_count=("local_authority", "count"))
    .reset_index()
    .sort_values("transaction_count", ascending=True)
)

return df
```

def create_chart(df: pd.DataFrame):
fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))

```
sns.barplot(
    data=df,
    x="transaction_count",
    y="local_authority",
    color="steelblue",
    ax=ax,
)

# Annotate bars with counts
for bar in ax.patches:
    ax.text(
        bar.get_width() + (df["transaction_count"].max() * 0.01),
        bar.get_y() + bar.get_height() / 2,
        f"{int(bar.get_width()):,}",
        va="center",
        fontsize=8,
    )

ax.set_title(
    "Transaction count by local authority",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Transaction Count", fontsize=11)
ax.set_ylabel("Local Authority", fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
return fig
```

def chart_pipeline(data):
slug = slugify.slugify(NAME)
df = data_pipeline(data)
fig = create_chart(df)
yield from matplotlib_output_generator(fig, f”{slug}-static”, “Static”)

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