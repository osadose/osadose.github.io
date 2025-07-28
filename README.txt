Hi there, I'm Ose



low_gdp_years = 0

for index, row in nigeria.iterrows():
    if row['gdpPercap'] < 1000:
        print(f"In {row['year']}, GDP per capita was low: ${row['gdpPercap']:.2f}")
        low_gdp_years += 1

print(f"\nTotal years with GDP per capita under $1,000: {low_gdp_years}")

