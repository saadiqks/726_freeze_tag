import random

f = open("input.txt", "w")

commands = ""
for i in range(1000):
    r = random.randint(1, 4)
    commands += f"{r} \n"

f.write(commands)
f.close()
