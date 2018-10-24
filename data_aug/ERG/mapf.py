from nltk.tokenize import word_tokenize
import os

sent = input()
info = input()

d = {}
m = {}
rm = {}

d["VB"] = ["ask", "assemble", "assess", "assign", "assume", "atone", "attention", "avoid", "bake", "balkanize", "bank", "begin", "behold", "believe", "bend", "benefit", "bevel", "beware", "bless", "boil", "bomb", "boost", "brace", "break", "bring", "broil", "brush", "build"]
d["VBZ"] = ["bases", "reconstructs", "marks", "mixes", "displeases", "seals", "carps", "weaves", "snatches", "slumps", "stretches", "authorizes", "smolders", "pictures", "emerges", "stockpiles", "seduces", "fizzes", "uses", "bolsters", "slaps", "speaks", "pleads"]
d["VBP"] = ["predominate", "wrap", "resort", "sue", "twist", "spill", "cure", "lengthen", "brush", "terminate", "appear", "tend", "stray", "glisten", "obtain", "comprise", "detest", "tease", "attract", "emphasize", "mold", "postpone", "sever", "return", "wag"]
d["VBD"] = ["dipped", "pleaded", "swiped", "regummed", "soaked", "tidied", "convened", "halted", "registered", "cushioned", "exacted", "snubbed", "strode", "aimed", "adopted", "belied", "figgered", "speculated", "wore", "appreciated", "contemplated"]
d["VBG"] = ["telegraphing", "stirring", "focusing", "angering", "judging", "stalling", "lactating", "alleging", "veering", "capping", "approaching", "traveling", "besieging", "encrypting", "interrupting", "erasing", "wincing"]
d["VBN"] = ["multihulled", "dilapidated", "aerosolized", "chaired", "languished", "panelized", "used", "experimented", "flourished", "imitated", "reunifed", "factored", "condensed", "sheared", "unsettled", "primed", "dubbed", "desired"]

d["FW"] = ["gemeinschaft", "hund", "ich", "jeux", "habeas", "vous", "lutihaw", "alai", "je", "jour", "objets", "salutaris", "fille", "quibusdam", "pas", "trop", "Monte", "terram", "fiche", "oui", "corporis"]

d["NN"] = ["cabbage", "shed", "thermostat", "investment", "slide", "humour", "falloff", "slick", "wind", "hyena", "override", "subhumanity", "machinist"]
d["NNS"] = ["undergraduates", "scotches", "products", "bodyguards", "facets", "coasts", "divestitures", "storehouses", "designs", "clubs", "fragrances", "averages", "subjectivists", "apprehensions", "muses"]
d["NNP"] = ["Motown", "Venneboerger", "Czestochwa", "Ranzer", "Conchita", "Trumplane", "Christos", "Oceanside", "Escobar", "Kreisler", "Sawyer", "Cougar", "Yvette", "Ervin", "ODI", "Darryl", "CTCA", "Shannon", "Meltex", "Liverpool"]
d["NNPS"] = ["Americans", "Americas", "Amharas", "Amityvilles", "Amusements", "Andalusians", "Andes", "Andruses", "Angels", "Animals", "Anthony", "Antilles", "Antiques", "Apache", "Apaches", "Apocrypha"]
d["JJ"] = ["third", "regrettable", "oiled", "calamitous", "first", "separable", "ectoplasmic", "participatory", "fourth", "multilingual"]
d["JJR"] = ["bleaker", "braver", "breezier", "briefer", "brighter", "brisker", "broader", "bumper", "busier", "calmer", "cheaper", "choosier", "cleaner", "clearer", "closer", "colder", "commoner", "costlier", "cozier", "creamier", "crunchier", "cuter"]
d["JJS"] = ["calmest", "cheapest", "choicest", "classiest", "cleanest", "clearest", "closest", "commonest", "corniest", "costliest", "crassest", "creepiest", "crudest", "cutest", "darkest", "deadliest", "dearest", "deepest", "densest", "dinkiest"]
d["CD"] = ["ten", "million", "0.5", "one", "1987", "twenty", "zero", "two", "78-degrees", "eighty-four", "IX", "", "fifteen", "271,124", "dozen", "quintillion", "DM2,000"]
d["RB"] = ["occasionally", "unabatingly", "maddeningly", "adventurously", "professedly", "stirringly", "prominently", "technologically", "magisterially", "predominately", "swiftly", "fiscally", "pitilessly"]


def get_tag(str, pos):
	e = i - 1
	while str[e] != '_':
		e = e - 1
	s = e - 1
	while str[s] != '/':
		s = s - 1
	return str[s + 1: e], s


def get_token(str, pos):
	s = pos - 1
	while str[s] != ' ':  # " _XXX/"
		s = s - 1
	return str[s + 2: pos]


def mapping(token, tag):
	if len(d[tag]) == 0:
		print("DICT_SIZE_ERROR")
		exit()
	x = d[tag][0]
	d[tag].remove(x)
	m[x] = token
	rm[token] = x


sent = word_tokenize(sent)
s = ""
for i in sent[2:]:
	if i in rm:
		s = s + rm[i] + " "
	else:
		s = s + i + " "

if sent[0] == 'SKIP':
	os.system("echo 0")
	os.system("echo \"" + s + "\"")
	exit()

for i in range(len(info) - 9):
	if info[i:i+9] == "u_unknown":
		tag, pos = get_tag(info, i)
		token = get_token(info, pos)

		mapping(token, tag)

os.system("echo " + str(len(m.items())) )
for i in m.items():
	os.system("echo " + i[0] + " " + i[1])

s = ""
for i in sent[2:]:
	if i in rm:
		s = s + rm[i] + " "
	else:
		s = s + i + " "

# print
os.system("echo \"" + s + "\"")
os.system("echo \"" + s + "\" | ./ace -g erg-1214-osx-0.9.27.dat -1T 2>/dev/null | ./ace -g erg-1214-osx-0.9.27.dat -e 2>/dev/null")
