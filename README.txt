Hi there, I'm Ose



low_gdp_years = 0

for index, row in nigeria.iterrows():
    if row['gdpPercap'] < 1000:
        print(f"In {row['year']}, GDP per capita was low: ${row['gdpPercap']:.2f}")
        low_gdp_years += 1

print(f"\nTotal years with GDP per capita under $1,000: {low_gdp_years}")

for i in range(3):  # Repeat 3 times
    print("Happy birthday to you")

print("Happy birthday dear Alice")
print("Happy birthday to you")

count = 0

while count < 4:
    if count == 2:
        print("Happy birthday dear Alice")
    else:
        print("Happy birthday to you")
    
    count += 1

language = input("Choose a language (english / spanish): ").lower()

if language == "english":
    print("Happy birthday to you")
    print("Happy birthday to you")
    print("Happy birthday dear Alice")
    print("Happy birthday to you")

elif language == "spanish":
    print("Cumpleaños feliz")
    print("Cumpleaños feliz")
    print("Te deseamos, Alice")
    print("Cumpleaños feliz")

else:
    print("No song for you! 😶")


