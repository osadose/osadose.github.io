Hi there, I'm Ose


---
title: "Geospatial analysis in Python"
format: 
  html:
    self-contained: true
jupyter: python3
---

## Learning objectives

-   read and write geospatial data
-   create a GeoDataFrame
-   transform CRS
-   run a spatial join
-   build a choropleth map with different classifications

## Aim

We will explore the distribution of public hospitals in the states of Nigeria by creating a choropleth map. Recent population projections are used to calculate a hospital density per million population.

## Data sources

-   [Natural Earth](https://www.naturalearthdata.com/)
-   [GRID3 NGA - Health Facilities v2.0](https://data.grid3.org/datasets/GRID3::grid3-nga-health-facilities-v2-0/about)
-   [NBS Demographic Statistics Bulletin 2021](https://www.nigerianstat.gov.ng/elibrary/read/1241207)

## Load libraries

```{python}
#| label: Load libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn
from mapclassify import EqualInterval, Quantiles, FisherJenks
import folium
```

## Read GeoJSON

```{python}
nigeria = gpd.read_file('data/nigeria.geojson')
nigeria.crs
```

## Plot data

```{python}
nigeria.plot()
```

```{python}
nigeria[nigeria['state']=='Edo'].explore()
```

## Read CSV

```{python}
hospitals = pd.read_csv('data/hospitals.csv')
hospitals.info()
```

## Convert to a GeoDataFrame

```{python}
hospitals_geo  = gpd.GeoDataFrame(hospitals, geometry=gpd.points_from_xy(hospitals.longitude, hospitals.latitude), crs = 'EPSG:4326')
hospitals_geo.plot()
```

## Check CRS match

```{python}
hospitals_geo.crs == nigeria.crs
```

## Plot layers

```{python}
m = nigeria.explore(tiles='CartoDB positron', 
  color='#ededed', 
  style_kwds={'fillOpacity':0.5, 'color':'black', 'opacity':0.7}
  )
hospitals_geo.explore(m=m, 
    color='#c51b8a', 
    style_kwds={'fillOpacity':1, 'color':'#FFFFFF', 'opacity':0.5}, 
    marker_kwds={'radius':5}
)
```

## Point in polyon

```{python}
hospitals_by_state = gpd.sjoin(hospitals_geo, nigeria, how = 'inner', predicate = 'within')
hospitals_by_state.head()
```

## Count hospitals per state

```{python}
hospitals_by_state_stats = hospitals_by_state.groupby(['state'])['state'].count().sort_values(ascending=False).reset_index(name='count')
hospitals_by_state_stats.sort_values('count', ascending=False).head()
```

# Merge datasets

```{python}
nigeria_hospitals = pd.merge(nigeria, hospitals_by_state_stats, on = 'state', how = 'left')
nigeria_hospitals.sort_values('count', ascending=False).head()
```

# Save results

``` python
nigeria_hospitals.to_file('data/nigeria_hospitals.geojson')
```

## Check distribution

```{python}
ax = seaborn.histplot(nigeria_hospitals["count"], bins=5)
seaborn.rugplot(nigeria_hospitals["count"], height=0.05, color="red", ax=ax);
nigeria_hospitals["count"].describe()
```

## Create choropleth map

```{python}
ax = nigeria_hospitals.plot(
    column='count',
    scheme='FisherJenks',
    cmap='YlGn',
    legend=True,
    legend_kwds={'title':'Number of hospitals', 'fmt': '{:.0f}', 'bbox_to_anchor':(1,0.28)},
)
nigeria.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.2)
ax.set_title("Hospitals by state in Nigeria")
ax.set_axis_off();
```

## Calculate rate

```{python}
population = pd.read_csv('data/population_by_state.csv')
nigeria_hospitals_rate = pd.merge(nigeria_hospitals, population, on = 'state', how = 'left')
nigeria_hospitals_rate['rate'] = (nigeria_hospitals_rate['count'] / nigeria_hospitals_rate['population'])*1000000
nigeria_hospitals_rate.sort_values('rate', ascending=False).head()
```

```{python}
ax = nigeria_hospitals_rate.plot(
    column='rate',
    scheme='FisherJenks',
    cmap='YlGn',
    legend=True,
    legend_kwds={'title':'Hospitals per\nmillion population', 'fmt': '{:.1f}', 'bbox_to_anchor':(1,0.28)},
)
nigeria.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.2)
ax.set_title("Hospital density by state in Nigeria")
ax.set_axis_off();
```
