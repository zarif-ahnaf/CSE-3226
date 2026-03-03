lst = []

i = 0

while i < 3:
    _lst = []
    j = 1
    while True:
        ask = input(
            f"Enter no {j} elements to enter to list {i + 1}. (give empty string to quit):"
        )
        if ask == "":
            break
        j += 1
        _lst.append(int(ask))
    print(f"List no {i + 1} is : {_lst}")
    lst.append(_lst)

    i += 1

print(f"Sum of all lists: {sum(sum(sublist) for sublist in lst)}")
