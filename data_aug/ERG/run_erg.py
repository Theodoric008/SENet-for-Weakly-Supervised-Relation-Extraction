import subprocess as sp
from IPython import embed

t = "a cat sat on the mat"

temp = sp.getoutput("echo \"" + t + "\" | ./ace -g erg-1214-osx-0.9.27.dat -1T 2>/dev/null | python3 mapf.py | python3 mapb.py")
tmp = temp.split("\n")
embed()
