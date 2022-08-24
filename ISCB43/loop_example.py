cost = 0.
grocery_items = ["apple", "celery", "bread"]
sale = False

for item in grocery_items:
    if item == "apple":
        cost = cost + 1.50
    if item == "celery":
        cost = cost + 3.50
    if item == "bread":
        if sale:
            cost = cost + (2.50*0.9)
        else:
            cost = cost + 2.50
