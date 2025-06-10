Hi there, I'm Ose


[
```markdown
---
title: "Week 5: Control Flow, Loops & Functions"
format: revealjs
theme: simple
logo: template/logo.png
---

## 1. Welcome & Objectives 🎯

**This week we cover:**
- 🧠 Conditionals (if/else)
- 🔁 Loop constructs (for, while)
- 🧩 Functions & code reuse

**Why this matters:**
Enable logic, iteration, and modularity in your Python programs.

---

## 2. Recap: What Came Before

- Weeks 0–4 covered Git, Data Structures, and Data Cleaning  
- You now know how to load and clean data—great work!  
- This week adds control structures and functions to build logic and reusable components

---

## 3. Control Flow: `if`, `elif`, `else`

```python
x = 10
if x < 0:
    print("Negative")
elif x == 0:
    print("Zero")
else:
    print("Positive")
```

- `if` checks a condition  
- `elif` for additional checks  
- `else` is the default fallback

---

## 4. Exercise Idea: Grade Checker

```python
score = 75
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D/F"
print(f"You got: {grade}")
```

Ask students: “What grade do you get for 85?”

---

## 5. Loops: `for` and `while`

**`for` loop** – iterate over a sequence

```python
for item in ["apple", "banana", "cherry"]:
    print(item)
```

**`while` loop** – repeat until condition is false

```python
count = 0
while count < 3:
    print(count)
    count += 1
```

---

## 6. Exercise Idea: Summation

```python
total = 0
for i in range(1, 11):
    total += i
print(total)  # expect 55
```

Great demonstration of accumulation!

---

## 7. Functions: Definitions & Usage

```python
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
```

- `def` keyword defines a function  
- Functions can `return` values  
- Reusable and modular

---

## 8. Exercise Idea: Custom Function

```python
def is_even(n):
    return n % 2 == 0

print(is_even(4))  # True
print(is_even(5))  # False
```

Ask students: “Write a function that checks if a number is prime.”

---

## 9. Putting It All Together 🧩

Combine control flow, loops, and functions:

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

for num in range(1, 21):
    print(num, is_prime(num))
```

✔️ Demonstrates key concepts together

---

## 10. This Week at a Glance

✅ You’ll be able to:
- Use `if`, `elif`, and `else`  
- Write `for` and `while` loops  
- Define and call functions  
- Combine them in real examples

🧭 In the live session:
- Instructor will *briefly walk through* these key slides  
- Then dive into hands-on practice with prompts and examples

---

