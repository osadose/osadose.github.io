Hi there, I'm Ose




def sing_birthday(language):
    if language == "english":
        print("Happy birthday to you")
    elif language == "spanish":
        print("Cumpleaños feliz")
    else:
        print("No song for you 😶")

# Run the function
sing_birthday("spanish")  # You can change to "english" or anything else

import pandas as pd

# Load and filter data
df = pd.read_csv("../../data/gapminder.csv")
nigeria = df.query("country == 'Nigeria'").copy()

# Classify GDP levels
nigeria['gdp_level'] = pd.cut(
    nigeria['gdpPercap'],
    bins=[0, 1500, 3000, float('inf')],
    labels=['low', 'medium', 'high'],
    right=False
)

# Calculate year-to-year GDP change
nigeria['gdp_change'] = nigeria['gdpPercap'].diff().round(2)

# Classify growth
nigeria['growth'] = pd.cut(
    nigeria['gdp_change'],
    bins=[-float('inf'), -0.01, 0.01, float('inf')],
    labels=['decrease', 'no change', 'increase']
)

# Select summary columns
summary = nigeria[['year', 'gdpPercap', 'gdp_level', 'gdp_change', 'growth']]
