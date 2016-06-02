from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from oct2py import octave
from oct2py.utils import Oct2PyError
import numpy.matlib
import numpy as np
import codecs
import nltk
import re
octave.addpath('src/octave')

datapath = 'data/'
docs = ['he', 'ro', 'ph', 'cl', 'ga', 'ep', 'co2', 'co', 'jo1', 'pe2', 'ja', 'pe1']
ids = ['Paul', 'John', 'Peter', 'James', 'Peter']

#unknown document
# f 'Hebrews'

#known documents
# f1 'Romans'
# f2 'Philippians'
# f3 'Colossians'
# f4 'Galatians'
# f5 'Ephesians'
# f6 '2 Corinthians'
# f7 '1 Corinthians'

#impostors
# f8 '1 John'
# f9 '2 Peter'
# f10 'James'
# f11 '1 Peter'

#convierte el vector de un texto individual a un vector basado en el vocabulario general del problema
def transformVec(strVec, numVec, vocabulary):
	newVec = []
	for w in vocabulary:
		if w in strVec:
			newVec.append(numVec[strVec.index(w)])
		else:
			newVec.append(0)
	return newVec

#concatena el contenido de una tupla en un solo string
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

#funcion delta que deja entradas en cero para vector x_0 excepto la i-esima
def delta(x_0, i):
 	x_i = np.matlib.zeros((len(x_0),1))
 	x_i[i] = x_0[i]
 	return x_i

def residual(dx,A,y):
	return np.linalg.norm(A*dx-y)

print "Who wrote the Epistle to the Hebrews?"
print "Is that you Paul?"

for f in docs:
	#leemos cada documento
	exec("file = codecs.open(datapath+'{0}.txt','r','utf-8')".format(f))
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
	exec("{0}_mc = contador.most_common(100)".format(f))
	file.close()
	exec("{0}_str = []".format(f))
	exec("{0}_num = []".format(f))
	exec("for w, n in {0}_mc:\n {0}_str.append(w)\n {0}_num.append(n)".format(f))
	#obtenemos bigramas
	big = ngrams(filtered_words, 2)
	bigCount = Counter(big)
	exec("{0}_big = bigCount.most_common(100)".format(f))
	exec("{0}_str_big = []".format(f))
	exec("{0}_num_big = []".format(f))
	exec("for w, n in {0}_big:\n {0}_str_big.append(w)\n {0}_num_big.append(n)".format(f))

#unimos todos los documentos de palabras en una lista para la representacion de bag of words
bowVec = he_str + ro_str + ph_str + cl_str + ga_str + ep_str + co2_str + co_str + jo1_str + pe2_str + ja_str + pe1_str

#unimos todos los documentos de palabras en una lista para la representacion de bigrams
bigVec = he_str_big + ro_str_big + ph_str_big + cl_str_big + ga_str_big + ep_str_big + co2_str_big + co_str_big + jo1_str_big + pe2_str_big + ja_str_big + pe1_str_big

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
paulSum = npro + npph + npcl + npga + npep + npco2 + npco
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

rows, cols = A_.shape
dx = np.array(rows)
r = 1000000000000000000000000000000000000
index = None

for i in range(cols):
	dx = delta(x_0, i)
	r_temp = residual(dx,A_,y_)
	print "Residual for {0}: {1}".format(ids[i],r_temp)
	if r_temp < r :
		index = i
		r = r_temp


print "Lowest residual: {0}".format(r)
print "It was you {0}".format(ids[index])


# trigrams = ngrams(filtered_words, 3)

# trigCount = Counter(trigrams)

# tM = trigCount.most_common(10)

# print '\n' + 'TRIGRAMS' + '\n'

# for i,j in tM:
# 	text = ''
# 	for x in i:
# 		text += x + ' '
# 	print '%s: %d' % (text, j)