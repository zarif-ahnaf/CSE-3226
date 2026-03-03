i = 1
arr = []

while i <= 3:
    ask = float(input(f"Enter your {i} number:"))
    arr.append(ask)
    i += 1

print(sum(arr))
