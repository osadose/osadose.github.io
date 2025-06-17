Hi there, I'm Ose



In this week's workshop, we will introduce version control using Git and also write our first basic Python scripts. Version control helps us track changes in code, collaborate with others, and avoid losing work. Python will be introduced as the main programming language for data science.

Exercise: Create your first Git repository and Python file
Your task is to:

Initialize a new Git repository.
Create a Python file named first_script.py.
In the Python file, write code to:
Print "Hello Nigeria!"
Create two variables: name (string), year (integer).
Write a simple function that returns a greeting message.
Commit your code and push to GitHub.

Further reading

VanderPlas, J. (2016). Python Data Science Handbook. O'Reilly.
ONS Data Science Campus (2021). Introduction to version control with Git.
Python Software Foundation. Official Python Documentation.



In this week's workshop, we will dive deeper into version control using Git. You will learn how to work with branches, resolve merge conflicts, and collaborate with others on a shared repository.

Lab

Exercise: Simulate a Git collaboration workflow
Your task is to:

Clone a shared repository provided by your instructor.
Create a new branch called feature-update.
Modify an existing Python file by adding a simple function (e.g. add two numbers).
Commit your changes and push the branch to GitHub.
Open a pull request.
Simulate a merge conflict by editing the same file in the main branch.
Resolve the conflict and merge your feature-update branch into main.


Further reading

O'Reilly: Loeliger, J., & McCullough, M. (2012). Version Control with Git, 2nd Edition. O'Reilly Media.
ONS Data Science Campus (2021). Introduction to version control with Git.
Git SCM. Official Git Documentation.




In this week's workshop, we will explore the core data types and data structures available in Python. Understanding how Python handles data is essential for effective programming, data analysis, and building complex applications.


Lab

Exercise: Practice with Python data types and structures
Your task is to:

Create variables of different types: integer, float, string, boolean.
Create:
A list of five Nigerian states.
A tuple with three currency codes.
A set of unique student IDs.
A dictionary mapping student names to scores.
Write functions to:
Print the length of the list.
Add a new state to the list.
Return all students with scores above 70 from the dictionary.
Test type conversion between strings and integers.
Commit and push your work to GitHub.


Further reading

O'Reilly: Beazley, D. (2023). Python Distilled. O'Reilly Media.
ONS Data Science Campus (2021). Introduction to Python for Data Science.
Python Software Foundation. Official Python Documentation - Data Types.


In this week's workshop, we will focus on importing data from various file formats and cleaning data to prepare it for analysis. Clean data is critical for any data science workflow. We will use the Pandas library for most of the tasks.

Lab

Exercise: Clean Gapminder Dataset
Your task is to clean and prepare the Gapminder dataset for analysis:

Load the gapminder.csv file.
Inspect the data using:
head(), info(), describe().
Rename columns:
For example, rename gdpPercap to GDP_per_capita.
Handle missing values:
Drop rows where lifeExp is missing.
Fill missing pop values with median population.
Filter the dataset to include only data for:
Nigeria, Ghana, Kenya, and South Africa.
Years 2000 and above.
Export the cleaned dataset to gapminder_cleaned.csv.
Commit and push your work to GitHub.


Further reading

O'Reilly: McKinney, W. (2022). Python for Data Analysis. O'Reilly Media.
ONS Data Science Campus (2021). Working with Data in Python.
Pandas Official Documentation: Working with Missing Data.

In this week's workshop, you will learn how to make your Python code more powerful and flexible using control flow, loops, and functions. This is the foundation of writing reusable and efficient Python code.

Lab

Exercise: Write functions to explore Gapminder dataset
Use the cleaned gapminder_cleaned.csv file from Week 4:

Write a function filter_country_year() that returns data for a specified country and year.
Write a function calculate_gdp() that adds a new column total_gdp by multiplying GDP_per_capita and pop.
Use a loop to calculate and print the average life expectancy for each country.
Use a loop to identify and print the country with the highest total GDP for each year.
Handle possible errors using try-except when country names are not found.
Save the final dataframe with total_gdp column to gapminder_final.csv.
Commit and push your code to GitHub.

Further reading

O'Reilly: Beazley, D. (2023). Python Distilled. O'Reilly Media.
ONS Data Science Campus (2021). Introduction to Python for Data Science.
Python Software Foundation. Python Control Flow.

In this week's workshop, we will explore Natural Language Processing (NLP) techniques for coding free-text survey responses, using the Nigeria Labour Force Survey (NBS) dataset.

Lab

Exercise: Clean and Code Labour Force Survey Free Text
You will work with the NBS Labour Force Survey 2023 dataset (nbs_labour_force_2023.csv) which contains free-text occupation responses:

Load the dataset and inspect the occupation_description column.
Clean the text:
Convert to lowercase.
Remove punctuation and numbers.
Remove stopwords.
Tokenize text and calculate most frequent words.
Create a simple occupation keyword dictionary that maps common terms to ISCO codes.
Write a function that assigns ISCO codes based on keyword matching.
Apply the function to the dataset and create a new column ISCO_code.
Export the updated dataset to labour_force_coded.csv.
Commit and push your work to GitHub.

Further reading

O'Reilly: Bird, S. et al. (2009). Natural Language Processing with Python (NLTK Book). O'Reilly Media.
ONS Data Science Campus (2022). Introduction to NLP in Python.
Nigeria National Bureau of Statistics: Labour Force Survey methodology documentation.
ISCO Official Classification: International Standard Classification of Occupations.








