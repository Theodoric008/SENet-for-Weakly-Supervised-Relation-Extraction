import random
from datetime import datetime

random.seed(datetime.now())

n = input()
d = {}

for i in range(int(n)):
	t = input()
	t = t.split(" ")
	d[t[0]] = t[1]

s = []

while True:
	try:
		sent = input()
		s.append(str(sent))
	except EOFError:
		break

for index in range(len(s)):
	sent = s[index].lower()
	print(sent)
