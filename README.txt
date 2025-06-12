Hi there, I'm Ose


---
title: "Week 6: Geospatial analysis in Python"
date: last-modified
format: 
  revealjs:
    theme:
      - default
    slide-number: true
    auto-stretch: false
    logo: logo.png
    embed-resources: true
scrollable: true
---

## Data models {.smaller}
:::: {.columns}

::: {.column width="50%"}
**Raster**    
- divides surface into regular cells    
- similar to digital images    
- e.g. remote sensing, elevation    
:::

::: {.column width="50%"}
**Vector**    
- features defined by x,y coordinates    
- classified into points, lines and polygons     
- e.g. administrative data    
:::

::::

## Maps distort reality {.smaller}
![](images/true_size_of_africa.jpg){width=45% fig-align="center"}

:::footer
Source: [The True Size of Africa](http://kai.sub.blue/images/True-Size-of-Africa-kk-v3.pdf)
:::

## Map projections {.smaller}

```{python}
#| label: fig-charts
#| layout-ncol: 2
#| fig-cap: 
#|   - "Mercator"
#|   - "Gall-Peters"

import geopandas as gpd
import matplotlib.pyplot as plt

world = gpd.read_file('data/ne_110m_admin_0_countries.shp')
world_sans_antarctica  = world[(world['ADMIN'] != "Antarctica")]

world_mercator = world_sans_antarctica .to_crs('EPSG:3395')
fig, ax = plt.subplots(figsize=(12,10))
world_mercator.plot(ax=ax, color="lightgray")
ax.set_axis_off()
plt.show()

world_peters = world_sans_antarctica .to_crs('ESRI:54002')
fig, ax = plt.subplots(figsize=(12,10))
world_peters.plot(ax=ax, color="lightgray")
ax.set_axis_off()
plt.show()
```

## Coordinate reference systems {.smaller}

::: columns
::: {.column width="50%"}
**Geographic CRS**    
![](images/vector_lonlat.png){width=50%}         
- position on the Earth's surface     
- degrees    
- latitude and longitude    
- position relative to the equator and the prime meridian   
- e.g. World Geodetic System (WGS84)   
:::

::: {.column width="50%"}
**Projected CRS**    
![](images/vector_projected.png){width=50%}    
- projects the surface of the earth onto a 2D plane    
- metres        
- x, y     
- e.g. British National Grid (BNG)  
:::
:::

:::footer
Images: [Dorman et al (2025)](https://py.geocompx.org/01-spatial-data)
:::

## Python  {.smaller}
We use Python's [GeoPandas](https://geopandas.org/en/stable) library to read and manipulate geospatial data.

Geospatial data is read as a `GeoDataFrame`, a regular [Pandas](https://pandas.pydata.org/) DataFrame with an additional `.geometry` column. The `.geometry` column contains the coordinates of each feature such as a state boundary. The attributes of each geometry (e.g. state name) are stored in other columns of the dataset.

For example, the following code uses the GeoPandas library to read an ESRI shapefile of the countries of the world, selects Nigeria and then prints out the first few rows.

```{.python}
import geopandas as gpd
world = gpd.read_file("data/ne_110m_admin_0_countries.shp")
nigeria = world[world['ADMIN'] == 'Nigeria']
nigeria.head()
```

## Further reading
- Dorman, M., Graser, A., Nowosad, J., & Lovelace, R. (2025). [Geocomputation with Python](https://py.geocompx.org/). CRC Press
- Rey, S.J., Arribas-Bel, D., & Wolf, L.J. (2023). [Geographic Data Science with Python](https://geographicdata.science/book/intro.html). CRC Press.
- Office for National Statistics Geospatial Team. (2021). [Introduction to GIS in Python](https://onsgeo.github.io/geospatial-training/docs/intro_to_gis_in_python)

