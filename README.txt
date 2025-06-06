Hi there, I'm Ose


[
```markdown
---
title: "Week 3: Data Types and Structures in Python"
format: revealjs
theme: simple
logo: template/logo.png
---

## 1. Welcome to Week 3! 🎉

**Theme:** Data Types and Structures in Python  
This week we’ll explore **how Python stores and organizes data**.

👨‍🏫 You’ll learn:
- The building blocks: strings, numbers, booleans
- How to use lists, tuples, dictionaries, and sets
- Real-world examples in data science

---

## 2. Why Data Types Matter 🧱

Every variable in Python has a **type** that determines:
- What kind of data it holds
- What operations you can perform

📊 Example:

```python
age = 25        # int
name = "Alice"  # str
is_valid = True # bool
```

Knowing the type helps you avoid errors and write better code.

---

## 3. Python's Built-in Data Types 🧠

Common types in Python:

| Type     | Example       | Purpose                        |
|----------|---------------|--------------------------------|
| `int`    | `10`          | Whole numbers                  |
| `float`  | `3.14`        | Decimal numbers                |
| `str`    | `"Hello"`     | Text data                      |
| `bool`   | `True/False`  | Logic/conditions               |

We use these constantly in Python scripts and data analysis.

---

## 4. Collections: Python's Data Structures 📦

Python has 4 main ways to **group multiple items**:

| Structure    | Mutable | Ordered | Use case                   |
|--------------|---------|---------|----------------------------|
| `list`       | ✅      | ✅      | Store items in order       |
| `tuple`      | ❌      | ✅      | Store fixed sequences      |
| `dict`       | ✅      | ✅      | Key-value pairs (like JSON)|
| `set`        | ✅      | ❌      | Unique items only          |

Each has its own strengths!

---

## 5. Lists: The Workhorse of Python 🛠️

```python
fruits = ["apple", "banana", "cherry"]
print(fruits[1])  # banana
```

✅ You can:
- Add: `fruits.append("orange")`
- Remove: `fruits.remove("apple")`
- Loop: `for f in fruits: print(f)`

Great for datasets and sequences!

---

## 6. Tuples and When to Use Them 📌

Tuples are like lists but **immutable** (unchangeable):

```python
coordinates = (10.0, 20.5)
```

✅ Use tuples for:
- Fixed values (e.g. coordinates, settings)
- Keys in dictionaries

🔒 You can’t `append()` or `remove()` from a tuple.

---

## 7. Dictionaries: Think in Key-Value Pairs 🗂️

```python
person = {
  "name": "Alice",
  "age": 25,
  "city": "Lagos"
}
```

Access with keys:

```python
print(person["name"])  # Alice
```

🧠 Ideal for:
- JSON-like data
- Mapping IDs to values
- Storing structured info

---

## 8. Sets: Unique & Unordered ✨

```python
colors = {"red", "green", "blue"}
```

✅ Use sets to:
- Remove duplicates
- Compare groups: `A & B`, `A | B`
- Check membership quickly

Example:

```python
"green" in colors  # True
```

---

## 9. Real-World Use in Data Science 🔬

Where you'll see data structures:

- CSV data → Lists of dictionaries
- JSON APIs → Dictionaries and nested lists
- Feature columns → Lists, arrays, sets
- Unique categories → Sets

Mastering types helps you **clean, explore, and analyze data confidently.**

---

## 10. What You’ll Do This Week 🏁

✅ Practice using:
- Strings, numbers, and booleans
- Lists, tuples, dicts, and sets
- Real-life data examples

🧪 Activities:
- Mini quizzes
- Hands-on exercises
- Refactoring messy data

Let’s get started!
```
]
