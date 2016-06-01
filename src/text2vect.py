from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from oct2py import octave
from oct2py.utils import Oct2PyError
import numpy as np
import codecs
import nltk
import re
octave.addpath('src/octave')

datapath = 'data/'
docs = {'he', 'ro', 'ph', 'cl', 'ga', 'ef', 'co2', 'co', 'jo1', 'pe2', 'ja', 'pe1'}

#unknown document
f = datapath + 'he.txt'

#known documents
f1 = datapath + 'ro.txt'
f2 = datapath + 'ph.txt'
f3 = datapath + 'cl.txt'
f4 = datapath + 'ga.txt'
f5 = datapath + 'ef.txt'
f6 = datapath + 'co2.txt'
f7 = datapath + 'co.txt'

#impostors
f8 = datapath + 'jo1.txt'
f9 = datapath + 'pe2.txt'
f10 = datapath + 'ja.txt'
f11 = datapath + 'pe1.txt'

#convierte el vector de un texto individual a un vector basado en el vocabulario general del problema
def transformVec(strVec, numVec, vocabulary):
	newVec = []
	for w in vocabulary:
		if w in strVec:
			newVec.append(numVec[strVec.index(w)])
		else:
			newVec.append(0)
	return newVec

def duple2Str(strVec):
	newVec = []
	a, b = zip(*strVec)
	for i in range(len(a)):
		c = a[i] + ' ' + b[i]
		newVec.append(c)
	return newVec

#elimina todas las palabras repetidas en el vocabulario
def createVocabulary(wordList):
	return set(wordList)

#leemos cada documento
file = codecs.open(f,'r','utf-8')
content = file.read()

#convertimos a minusculas
content = content.lower()
#quitamos numeros y signos de puntuacion
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
#quitamos palabras funcionales
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
#obtenemos palabras mas comunes
he_mc = contador.most_common(100)
file.close()
he_str = []
he_num = []
for w, n in he_mc:
	he_str.append(w)
	he_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
he_big = bigCount.most_common(100)
he_str_big = []
he_num_big = []
for w, n in he_big:
	he_str_big.append(w)
	he_num_big.append(n)



file = codecs.open(f1,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
ro_mc = contador.most_common(100)
file.close()
ro_str = []
ro_num = []
for w, n in ro_mc:
	ro_str.append(w)
	ro_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
ro_big = bigCount.most_common(100)
ro_str_big = []
ro_num_big = []
for w, n in ro_big:
	ro_str_big.append(w)
	ro_num_big.append(n)



file = codecs.open(f2,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
ph_mc = contador.most_common(100)
file.close()
ph_str = []
ph_num = []
for w, n in ph_mc:
	ph_str.append(w)
	ph_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
ph_big = bigCount.most_common(100)
ph_str_big = []
ph_num_big = []
for w, n in ph_big:
	ph_str_big.append(w)
	ph_num_big.append(n)


file = codecs.open(f3,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
cl_mc = contador.most_common(100)
file.close()
cl_str = []
cl_num = []
for w, n in cl_mc:
	cl_str.append(w)
	cl_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
cl_big = bigCount.most_common(100)
cl_str_big = []
cl_num_big = []
for w, n in cl_big:
	cl_str_big.append(w)
	cl_num_big.append(n)


file = codecs.open(f4,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
ga_mc = contador.most_common(100)
file.close()
ga_str = []
ga_num = []
for w, n in ga_mc:
	ga_str.append(w)
	ga_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
ga_big = bigCount.most_common(100)
ga_str_big = []
ga_num_big = []
for w, n in ga_big:
	ga_str_big.append(w)
	ga_num_big.append(n)


file = codecs.open(f5,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
ef_mc = contador.most_common(100)
file.close()
ef_str = []
ef_num = []
for w, n in ef_mc:
	ef_str.append(w)
	ef_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
ef_big = bigCount.most_common(100)
ef_str_big = []
ef_num_big = []
for w, n in ef_big:
	ef_str_big.append(w)
	ef_num_big.append(n)


file = codecs.open(f6,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
co2_mc = contador.most_common(100)
file.close()
co2_str = []
co2_num = []
for w, n in co2_mc:
	co2_str.append(w)
	co2_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
co2_big = bigCount.most_common(100)
co2_str_big = []
co2_num_big = []
for w, n in co2_big:
	co2_str_big.append(w)
	co2_num_big.append(n)


file = codecs.open(f7,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
co_mc = contador.most_common(100)
file.close()
co_str = []
co_num = []
for w, n in co_mc:
	co_str.append(w)
	co_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
co_big = bigCount.most_common(100)
co_str_big = []
co_num_big = []
for w, n in co_big:
	co_str_big.append(w)
	co_num_big.append(n)


file = codecs.open(f8,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
jo1_mc = contador.most_common(100)
file.close()
jo1_str = []
jo1_num = []
for w, n in jo1_mc:
	jo1_str.append(w)
	jo1_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
jo1_big = bigCount.most_common(100)
jo1_str_big = []
jo1_num_big = []
for w, n in jo1_big:
	jo1_str_big.append(w)
	jo1_num_big.append(n)



file = codecs.open(f9,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
pe2_mc = contador.most_common(100)
file.close()
pe2_str = []
pe2_num = []
for w, n in pe2_mc:
	pe2_str.append(w)
	pe2_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
pe2_big = bigCount.most_common(100)
pe2_str_big = []
pe2_num_big = []
for w, n in pe2_big:
	pe2_str_big.append(w)
	pe2_num_big.append(n)


file = codecs.open(f10,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
ja_mc = contador.most_common(100)
file.close()
ja_str = []
ja_num = []
for w, n in ja_mc:
	ja_str.append(w)
	ja_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
ja_big = bigCount.most_common(100)
ja_str_big = []
ja_num_big = []
for w, n in ja_big:
	ja_str_big.append(w)
	ja_num_big.append(n)


file = codecs.open(f11,'r','utf-8')
content = file.read()
content = content.lower()
toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
nc = toker.tokenize(content)
filtered_words = [w for w in nc if not w in stopwords.words('english')]
contador = Counter(filtered_words)
pe1_mc = contador.most_common(100)
file.close()
pe1_str = []
pe1_num = []
for w, n in pe1_mc:
	pe1_str.append(w)
	pe1_num.append(n)
#obtenemos bigramas
big = ngrams(filtered_words, 2)
bigCount = Counter(big)
pe1_big = bigCount.most_common(100)
pe1_str_big = []
pe1_num_big = []
for w, n in pe1_big:
	pe1_str_big.append(w)
	pe1_num_big.append(n)

#unimos todos los documentos de palabras en una lista para la representacion de bag of words
bowVec = he_str + ro_str + ph_str + cl_str + ga_str + ef_str + co2_str + co_str + jo1_str + pe2_str + ja_str + pe1_str

#unimos todos los documentos de palabras en una lista para la representacion de bigrams
bigVec = he_str_big + ro_str_big + ph_str_big + cl_str_big + ga_str_big + ef_str_big + co2_str_big + co_str_big + jo1_str_big + pe2_str_big + ja_str_big + pe1_str_big

#creamos el vocabulario para bag of words
bowVoc = createVocabulary(bowVec)

#creamos el vocabulario para bigrams
tempVoc = duple2Str(bigVec) #transformamos la tupla de palabras concatenando ambas palabras en una
bigVoc = createVocabulary(tempVoc)

#convertimos cada documento en un vector basado en el vocabulario de bag of words
for elem in docs:
	exec("global new{0}; new{0} = transformVec({0}_str, {0}_num, bowVoc)".format(elem))

#convertimos cada documentos en un vector basado en el vocabulario de bigramas

for elem in docs:
	#transformamos la tupla de palabras concatenando ambas palabras en una
	exec("global d2s{0}; d2s{0} = duple2Str({0}_str_big)".format(elem))
	exec("global new_big{0}; new_big{0} = transformVec(d2s{0}, {0}_num_big, bigVoc)".format(elem))

#concatenamos las representaciones en una sola
for elem in docs:
	exec("global finalVec{0}; finalVec{0} = new{0}+new_big{0}".format(elem))

for elem in docs:
	exec("global np{0}; np{0} = np.array(finalVec{0})".format(elem))


#calculamos el vector representativo de Pablo
paulSum = npro + npph + npcl + npga + npef + npco2 + npco
paulAvg = paulSum/7

#resolvemos con homotopia
nu=0.0001
tol=0.0001
stopCrit=3

A = np.matrix([paulAvg, npjo1, nppe2, npja, nppe1])
A_ = A.T
y = np.matrix(nphe)
y_ = y.T

try:
	x_0, nIter = octave.SolveHomotopy(A_, y_, 'lambda', nu, 'tolerance', tol, 'stoppingcriterion', stopCrit)
except Oct2PyError:
	pass

#imprimimos el vector disperso
print x_0

#for column in A:


#bigrams = ngrams(filtered_words, 2)

# bigCount = Counter(bigrams)

# bM = bigCount.most_common(10)

# print '\n' + 'BIGRAMS' + '\n'

# for i,j in bM:
# 	text = ''
# 	for x in i:
# 		text += x + ' '
# 	print '%s: %d' % (text, j)


# trigrams = ngrams(filtered_words, 3)

# trigCount = Counter(trigrams)

# tM = trigCount.most_common(10)

# print '\n' + 'TRIGRAMS' + '\n'

# for i,j in tM:
# 	text = ''
# 	for x in i:
# 		text += x + ' '
# 	print '%s: %d' % (text, j)