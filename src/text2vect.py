from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize.punkt import PunktLanguageVars
from cltk.stop.greek.stops import STOPS_LIST
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams
from oct2py import octave
import matplotlib.pyplot as plt
from oct2py.utils import Oct2PyError
import numpy.matlib
import numpy as np
import argparse
import codecs
import nltk
import re
octave.addpath('src/octave')


#documentos y sus identificadores que se cargaran y procesaran como vectores
docs = ['he', 'ro', 'ph', 'cl', 'ga', 'ep', 'co2', 'co', 'jo1','jo2', 'jo3', 'pe2', 'ja', 'pe1', 'ju', 'ma', 'mr', 'lu', 'jn', 'ac', 're']
#documentos utilizados para crear la matriz con la que se resolvera el sistema
docsMatrix = ['ro', 'ph', 'cl', 'ga', 'ep', 'co2', 'co', 'jo1','jo2', 'jo3', 'pe2', 'ja', 'pe1', 'ju', 'ma', 'mr', 'lu', 'jn', 'ac', 're']
#identidad de cada autor en el mismo orden que en la lista docsMatrix
ids = ['Paul', 'Paul', 'Paul', 'Paul', 'Paul', 'Paul', 'Paul','John','John','John', 'Peter', 'James', 'Peter', 'Judas', 'Matthew', 'Mark', 'Luke', 'John', 'Paul', 'John']
#representaciones utilizadas
reps = ['bow', 'big', 'trig', 'punct', 'stem', 'pos']
#vector de ceros para iniciar conteo de autores que contribuyen mas en cada iteracion
counter = np.zeros(len(ids))

#Numero de elementos mas comunes que se utilizaran para cada atributo: bag of words, bigramas, trigramas...
#num_common = 20

#iteraciones de todo el algoritmo
iterations = 200

#contador casos validos para SCI
sciValidIter=0


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
# f15 'Matthew'
# f16 'Mark'
# f17 'Luke'
# f18 'John'
# f19 'Paul'
# f20 'John'

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
def tuple2Str_big(strVec):
	newVec = []
	a, b = zip(*strVec)
	for i in range(len(a)):
		c = a[i] + ' ' + b[i]
		newVec.append(c)
	return newVec

#concatena el contenido de una tupla en un solo string para trigramas
def tuple2Str_trig(strVec):
	newVec = []
	a, b, c = zip(*strVec)
	for i in range(len(a)):
		d = a[i] + ' ' + b[i] + ' ' + c[i]
		newVec.append(d)
	return newVec

def choosePOS(posList):
	return [y for x, y in posList]

#elimina todas las palabras repetidas en el vocabulario
def createVocabulary(wordList):
	return list(set(wordList))

#funcion delta que deja entradas en cero para vector x_0 excepto la i-esima
def delta(x_0, i):
 	x_i = np.matlib.zeros((len(x_0),1))
 	x_i[i] = x_0[i]
 	return x_i

#calculamos residual de la operacion || y - A * delta(x) ||_2
def residual(dx,A,y):
	return np.linalg.norm(y-A*dx)

#indice de concentracion de dispersion donde i son los coeficientes de x y k es el num de identidades
#en nuestro caso podemos decir que son el mismo numero
#y cols es el numero de columnas en nuestra matriz A
def sci(x, i, k, cols):
    return (k*max_delta(x, cols)-1)/(k-1)

#escoge la norma l1 mas grande de todas los xi al aplicar delta para cada coeficiente en x
def max_delta(x, cols):
    norml1 = [(np.linalg.norm(delta(x,i),1)/np.linalg.norm(x,1)) for i in range(cols)]
    return np.max(norml1)

#________________Aqui comienza el programa

print("Who wrote the Epistle to the Hebrews?")
print("Is that you Paul?")
#language = 'english'
language = input("Enter desired language: (english or greek) ") #raw_input para python2 
datapath = 'data/' + language + '/'

for i in range(iterations):
    num_common=i+1
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

    	p = PunktLanguageVars()
    	ncGreek = p.word_tokenize(content)

    	#quitamos palabras funcionales
    	if language=='english':
    		filtered_words = [w for w in nc if not w in stopwords.words(language)]
    	elif language=='greek':
    		filtered_words = [w for w in ncGreek if not w in STOPS_LIST]

    	#creamos un diccionario y contamos los elementos mas comunes para bag of words, bigramas y trigramas
    	contador = Counter(filtered_words)	

    	#obtenemos palabras mas comunes
    	exec("{0}_mc = contador.most_common(num_common)".format(f))
    	exec("{0}_str_bow = []".format(f))
    	exec("{0}_num_bow = []".format(f))
    	exec("for w, n in {0}_mc:\n {0}_str_bow.append(w)\n {0}_num_bow.append(n)".format(f))
    	
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
    	contadorPunct = Counter(ncPunct)
    	exec("{0}_punct = contadorPunct.most_common(num_common)".format(f))
    	exec("{0}_str_punct = []".format(f))
    	exec("{0}_num_punct = []".format(f))
    	exec("for w, n in {0}_punct:\n {0}_str_punct.append(w)\n {0}_num_punct.append(n)".format(f))
    	
    	#obtenemos prefijos con el detalle de que primero obtenemos prefijos y contamos los mas comunes
    	st = LancasterStemmer()
    	exec("{0}_stemTemp = [st.stem(item) for item in filtered_words]".format(f))
    	exec("contadorStem = Counter({0}_stemTemp)".format(f))
    	exec("{0}_stem = contadorStem.most_common(num_common)".format(f))
    	exec("{0}_str_stem = []".format(f))
    	exec("{0}_num_stem = []".format(f))
    	exec("for w, n in {0}_stem:\n {0}_str_stem.append(w)\n {0}_num_stem.append(n)".format(f))

    	#obtenemos etiquetas pos 
    	pos = pos_tag(nc)
    	pos_1 = choosePOS(pos)
    	contadorPos = Counter(pos_1)
    	exec("{0}_pos = contadorPos.most_common(num_common)".format(f))
    	exec("{0}_str_pos = []".format(f))
    	exec("{0}_num_pos = []".format(f))
    	exec("for w, n in {0}_pos:\n {0}_str_pos.append(w)\n {0}_num_pos.append(n)".format(f))

    #unimos todos los documentos de palabras en una lista para la representacion de bag of words
    bowVec = []
    for elem in docs:
    	exec("bowVec+={0}_str_bow".format(elem))

    #unimos todos los documentos de palabras en una lista para la representacion de bigramas
    bigVec = []
    for elem in docs:
    	exec("bigVec+={0}_str_big".format(elem))

    #unimos todos los documentos de palabras en una lista para la representacion de trigramas
    trigVec = []
    for elem in docs:
    	exec("trigVec+={0}_str_trig".format(elem))

    #unimos todos los documentos de palabras en una lista para la representacion de puntuacion
    punctVec = []
    for elem in docs:
    	exec("punctVec+={0}_str_punct".format(elem))

    #unimos todos los documentos de palabras en una lista para la representacion de prefijos
    stemVec = []
    for elem in docs:
    	exec("stemVec+={0}_str_stem".format(elem))

    #unimos todos los documentos de palabras en una lista para la representacion pos
    posVec = []
    for elem in docs:
    	exec("posVec+={0}_str_pos".format(elem))

    #creamos vocabulario general
    for rep in reps:
    	if rep == 'big' or rep =='trig':
    		#transformamos la tupla de palabras concatenando ambas palabras en una
    		exec("tempVoc = tuple2Str_{0}({0}Vec)".format(rep))
    		#creamos el vocabulario para bigramas o trigramas
    		exec("{0}Voc=createVocabulary({0}Vec)".format(rep))
    	else:
    		#creamos el vocabulario para el resto de las representaciones
    		exec("{0}Voc=createVocabulary({0}Vec)".format(rep))

    #proyeccion de documentos en vectores

    #convertimos cada documento en un vector basado en el vocabulario de bag of words
    for elem in docs:
    	exec("global new_bow{0}; new_bow{0} = transformVec({0}_str_bow, {0}_num_bow, bowVoc)".format(elem))

    #convertimos cada documento en un vector basado en el vocabulario de bigramas
    for elem in docs:
    	#transformamos la tupla de palabras concatenando ambas palabras en una
    	exec("global d2s{0}; d2s{0} = tuple2Str_big({0}_str_big)".format(elem))
    	exec("global new_big{0}; new_big{0} = transformVec(d2s{0}, {0}_num_big, bigVoc)".format(elem))

    #convertimos cada documentos en un vector basado en el vocabulario de trigramas
    for elem in docs:
    	#transformamos la tupla de palabras concatenando ambas palabras en una
    	exec("global d2s{0}; d2s{0} = tuple2Str_trig({0}_str_trig)".format(elem))
    	exec("global new_trig{0}; new_trig{0} = transformVec(d2s{0}, {0}_num_trig, trigVoc)".format(elem))

    #convertimos cada documento en un vector basado en el vocabulario de puntuacion
    for elem in docs:
    	exec("global new_punct{0}; new_punct{0} = transformVec({0}_str_punct, {0}_num_punct, punctVoc)".format(elem))

    #convertimos cada documento en un vector basado en el vocabulario de prefijos
    for elem in docs:
    	exec("global new_stem{0}; new_stem{0} = transformVec({0}_str_stem, {0}_num_stem, stemVoc)".format(elem))

    #convertimos cada documento en un vector basado en el vocabulario de pos tags
    for elem in docs:
    	exec("global new_pos{0}; new_pos{0} = transformVec({0}_str_pos, {0}_num_pos, posVoc)".format(elem))


    #concatenamos las representaciones en una sola
    for elem in docs:
    	full_rep = []
    	separator="+"
    	for rep in reps:
    		full_rep.append("new_{1}{0}".format(elem,rep))
    	full_rep_str = separator.join(full_rep)
    	exec("global finalVec{0}; finalVec{0} = {1}".format(elem, full_rep_str))

    #creamos np arrays para los vectores finales de cada documento para poder convertirlos en una matriz y operar con ellos
    for elem in docs:
    	exec("global np{0}; np{0} = np.array(finalVec{0})".format(elem))

    #calculamos el vector representativo de Pablo
    #aunque no lo utilizamos hasta que haya una forma de darle un peso adecuado
    #paulSum = npro + npph + npcl + npga + npep + npco2 + npco
    #paulAvg = paulSum

    #resolvemos con homotopia
    nu=0.0001
    tol=0.0001
    stopCrit=3

    #creamos la lista de los vectores para crear la matriz A
    matrixElem = []
    for elem in docsMatrix:
    	exec("matrixElem.append(np{0})".format(elem))

    #Construimos el sistema de ecuaciones Ax=y

    A = np.matrix(matrixElem)
    A_ = A.T
    #asignamos a y el documento desconocido nphe que en este caso es el documento de Hebreos (he)
    y = np.matrix([nphe])
    y_ = y.T

    #resolvemos el sistema utilizando homotopia
    try:
    	x_0, nIter = octave.SolveHomotopy(A_, y_, 'lambda', nu, 'tolerance', tol, 'stoppingcriterion', stopCrit)
    except Oct2PyError:
    	pass

    #graficamos el vector disperso resultante
    # y_pos = np.arange(len(ids))
    # plt.bar(y_pos, x_0, align='center', alpha=0.5)
    # plt.xticks(y_pos, ids)
    # plt.ylabel('Contribution')
    # plt.title('Sparse Vector X')
    # plt.show()

    rows, cols = A_.shape
    dx = np.array(rows)
    r = 1000000000
    tau = 0.25 #25
    numIds = len(ids)
    residuales= []

    #obtenemos el residual mas pequeno de entre todos los vectores columna de A
    for i in range(cols):
        #calculamos delta de x_0 para cada coeficiente i
        dx = delta(x_0, i)
        #calculamos residual de d(x_0)
        residuales.append(residual(dx,A_,y_))
        #print("Residual for {0}: {1}".format(ids[i],r_temp))

    #buscamos el residual mas pequeno
    authorId = np.argmin(residuales)

    #calculamos sci para ver si x recuperado es valido o no
    if sci(x_0, numIds, numIds, cols)>= tau:
        counter[authorId]+=1
        sciValidIter+=1
        #print(sci(x_0, numIds, numIds, cols))
        
        #graficamos el vector disperso resultante
        #y_pos = np.arange(len(ids))
        #plt.bar(y_pos, x_0, align='center', alpha=0.5)
        #plt.xticks(y_pos, ids)
        #plt.ylabel('Contribution')
        #plt.title('Sparse Vector X')
        #plt.show()
    

print("Possible Author: {0}".format(ids[np.argmax(counter)]))
print("Probability: {0}%".format((counter[np.argmax(counter)]/sciValidIter)*100))

for i in range(len(counter)):
    print("{0}: {1}%".format(ids[i],(counter[i]/sciValidIter)*100))


