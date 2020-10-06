import datetime


now = datetime.datetime.now().time()
print(now.hour)
print(now.minute)
if now.hour == 14:
    print("found 14")
if now.minute == 24:
    print("found 15")

