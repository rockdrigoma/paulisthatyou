from nltk import sent_tokenize, word_tokenize, pos_tag
from cltk.stop.greek.stops import STOPS_LIST
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from oct2py import octave
from oct2py.utils import Oct2PyError
import argparse
import numpy.matlib
import numpy as np
import codecs
import nltk
import re
octave.addpath('src/octave')

docs = ['he', 'ro', 'ph', 'cl', 'ga', 'ep', 'co2', 'co', 'jo1','jo2', 'jo3', 'pe2', 'ja', 'pe1', 'ju']
ids = ['Paul', 'Paul', 'Paul', 'Paul', 'Paul', 'Paul', 'Paul','John','John','John', 'Peter', 'James', 'Peter', 'Judas']
num_common = 10

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
# f9 '2 John'
# f10 '3 John'
# f11 '2 Peter'
# f12 'James'
# f13 '1 Peter'
# f14 'Judas'

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
def tuple2Str(strVec):
	newVec = []
	a, b = zip(*strVec)
	for i in range(len(a)):
		c = a[i] + ' ' + b[i]
		newVec.append(c)
	return newVec

#concatena el contenido de una tupla en un solo string para trigramas
def tuple2StrTrig(strVec):
	newVec = []
	a, b, c = zip(*strVec)
	for i in range(len(a)):
		d = a[i] + ' ' + b[i] + ' ' + c[i]
		newVec.append(d)
	return newVec

#elimina todas las palabras repetidas en el vocabulario
def createVocabulary(wordList):
	return list(set(wordList))

#funcion delta que deja entradas en cero para vector x_0 excepto la i-esima
def delta(x_0, i):
 	x_i = np.matlib.zeros((len(x_0),1))
 	x_i[i] = x_0[i]
 	return x_i

def residual(dx,A,y):
	return np.linalg.norm(A*dx-y)

#Aqui comienza el programa

print("Who wrote the Epistle to the Hebrews?")
print("Is that you Paul?")
language = input("Enter desired language: (english or greek) ") #raw_input para python2 
datapath = 'data/' + language + '/'

for f in docs:
	#leemos cada documento
	exec("file = codecs.open(datapath+'{0}.txt','r','utf-8')".format(f))
	content = file.read()
	file.close()

	#convertimos a minusculas
	content = content.lower()
	#quitamos numeros y signos de puntuacion para bag of words, bigramas y trigramas
	toker = RegexpTokenizer(r'\W+|(,.;)+|[0-9]+', gaps=True)
	nc = toker.tokenize(content)
	#dejamos solo puntuacion para representacion de signos de puntuacion
	tokerPunct = RegexpTokenizer(r'[^,.;!?]+', gaps=True)
	ncPunct = tokerPunct.tokenize(content)

	#quitamos palabras funcionales
	if language=='english':
		filtered_words = [w for w in nc if not w in stopwords.words(language)]
	elif language=='greek':
		filtered_words = [w for w in nc if not w in STOPS_LIST]

	#creamos un diccionario y contamos los elementos mas comunes para bag of words, bigramas y trigramas
	contador = Counter(filtered_words)
	#creamos un diccionario y contamos los signos de puntuacion mas comunes
	contadorPunct = Counter(ncPunct)

	#obtenemos palabras mas comunes
	exec("{0}_mc = contador.most_common(num_common)".format(f))
	exec("{0}_str = []".format(f))
	exec("{0}_num = []".format(f))
	exec("for w, n in {0}_mc:\n {0}_str.append(w)\n {0}_num.append(n)".format(f))
	#obtenemos bigramas
	big = ngrams(filtered_words, 2)
	bigCount = Counter(big)
	exec("{0}_big = bigCount.most_common(num_common)".format(f))
	exec("{0}_str_big = []".format(f))
	exec("{0}_num_big = []".format(f))
	exec("for w, n in {0}_big:\n {0}_str_big.append(w)\n {0}_num_big.append(n)".format(f))
	#obtenemos trigramas
	trig = ngrams(filtered_words, 3)
	trigCount = Counter(trig)
	exec("{0}_trig = trigCount.most_common(num_common)".format(f))
	exec("{0}_str_trig = []".format(f))
	exec("{0}_num_trig = []".format(f))
	exec("for w, n in {0}_trig:\n {0}_str_trig.append(w)\n {0}_num_trig.append(n)".format(f))
	#obtenemos signos de puntuacion
	exec("{0}_punct = contadorPunct.most_common(num_common)".format(f))
	exec("{0}_str_punct = []".format(f))
	exec("{0}_num_punct = []".format(f))
	exec("for w, n in {0}_punct:\n {0}_str_punct.append(w)\n {0}_num_punct.append(n)".format(f))

#unimos todos los documentos de palabras en una lista para la representacion de bag of words
bowVec = he_str + ro_str + ph_str + cl_str + ga_str + ep_str + co2_str + co_str + jo1_str + jo2_str + jo3_str + pe2_str + ja_str + pe1_str + ju_str

#unimos todos los documentos de palabras en una lista para la representacion de bigramas
bigVec = he_str_big + ro_str_big + ph_str_big + cl_str_big + ga_str_big + ep_str_big + co2_str_big + co_str_big + jo1_str_big + jo2_str_big + jo3_str_big + pe2_str_big + ja_str_big + pe1_str_big + ju_str_big

#unimos todos los documentos de palabras en una lista para la representacion de trigramas
trigVec = he_str_trig + ro_str_trig + ph_str_trig + cl_str_trig + ga_str_trig + ep_str_trig + co2_str_trig + co_str_trig + jo1_str_trig + jo2_str_trig + jo3_str_trig + pe2_str_trig + ja_str_trig + pe1_str_trig + ju_str_trig

#unimos todos los documentos de palabras en una lista para la representacion de puntuacion
punctVec = he_str_punct + ro_str_punct + ph_str_punct + cl_str_punct + ga_str_punct + ep_str_punct + co2_str_punct + co_str_punct + jo1_str_punct + jo2_str_punct + jo3_str_punct + pe2_str_punct + ja_str_punct + pe1_str_punct + ju_str_punct

#creamos el vocabulario para bag of words
bowVoc = createVocabulary(bowVec)
#creamos el vocabulario para bigrams
tempVoc = tuple2Str(bigVec) #transformamos la tupla de palabras concatenando ambas palabras en una
bigVoc = createVocabulary(tempVoc)

#creamos el vocabulario para trigrams
tempVoc = tuple2StrTrig(trigVec) #transformamos la tupla de palabras concatenando ambas palabras en una
trigVoc = createVocabulary(tempVoc)

#creamos el vocabulario para puntuacion
punctVoc = createVocabulary(punctVec)

#convertimos cada documento en un vector basado en el vocabulario de bag of words
for elem in docs:
	exec("global new{0}; new{0} = transformVec({0}_str, {0}_num, bowVoc)".format(elem))

#convertimos cada documentos en un vector basado en el vocabulario de bigramas
for elem in docs:
	#transformamos la tupla de palabras concatenando ambas palabras en una
	exec("global d2s{0}; d2s{0} = tuple2Str({0}_str_big)".format(elem))
	exec("global new_big{0}; new_big{0} = transformVec(d2s{0}, {0}_num_big, bigVoc)".format(elem))

#convertimos cada documentos en un vector basado en el vocabulario de trigramas
for elem in docs:
	#transformamos la tupla de palabras concatenando ambas palabras en una
	exec("global d2s{0}; d2s{0} = tuple2StrTrig({0}_str_trig)".format(elem))
	exec("global new_trig{0}; new_trig{0} = transformVec(d2s{0}, {0}_num_trig, trigVoc)".format(elem))

#convertimos cada documento en un vector basado en el vocabulario de puntuacion
for elem in docs:
	exec("global new_punct{0}; new_punct{0} = transformVec({0}_str_punct, {0}_num_punct, punctVoc)".format(elem))

#concatenamos las representaciones en una sola
for elem in docs:
	exec("global finalVec{0}; finalVec{0} = new{0}+new_big{0}+new_trig{0}+new_punct{0}".format(elem))

#creamos np arrays para los vectores finales de cada documento para poder convertirlos en una matriz y operar con ellos
for elem in docs:
	exec("global np{0}; np{0} = np.array(finalVec{0})".format(elem))


#calculamos el vector representativo de Pablo, aunque no lo utilizamos hasta que haya una forma de darle un peso adecuado
paulSum = npro + npph + npcl + npga + npep + npco2 + npco
paulAvg = paulSum

#resolvemos con homotopia
nu=0.0001
tol=0.0001
stopCrit=3

A = np.matrix([npro, npph, npcl, npga, npep, npco2, npco, npjo1, npjo2, npjo3, nppe2, npja, nppe1, npju])
A_ = A.T
y = np.matrix(nphe)
y_ = y.T

try:
	x_0, nIter = octave.SolveHomotopy(A_, y_, 'lambda', nu, 'tolerance', tol, 'stoppingcriterion', stopCrit)
except Oct2PyError:
	pass

rows, cols = A_.shape
dx = np.array(rows)
r = 1000000000
authorIndex = None

for i in range(cols):
	dx = delta(x_0, i)
	r_temp = residual(dx,A_,y_)
	print("Residual for {0}: {1}".format(ids[i],r_temp))
	if r_temp < r :
		authorIndex = i
		r = r_temp

print("Lowest residual: {0}".format(r))
print("It was you {0}".format(ids[authorIndex]))