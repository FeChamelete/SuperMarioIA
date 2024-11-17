import sys
import retro
import pickle

import numpy as np
from numpy.random import uniform, choice, random
import pickle
import os
import time
from rominfo import *
#from utils import *

# Todas as possíveis ações
actions_map = {'noop':0, 'down':32, 'up':16, 'jump':1, 'spin':3, 
               'left':64, 'jumpleft':65, 'runleft':66, 'runjumpleft':67, 
               'right':128, 'jumpright':129, 'runright':130, 'runjumpright':131, 
               'spin':256, 'spinright':384, 'runspinright':386, 'spinleft':320, 'spinrunleft':322
               }

# Vamos usar apenas um subconjunto
#actions_list = [66,130,128,131,386]
#actions_list = [1,128,130,131,256,386]
qtdAcoes      = 8
actions_list  = [0,1,64,128,129,130,131,386]

radius = 5

qTable = {} #dicionario usa hashtable, mais rapida a busca dos estados
qTableContador = {}

#faz a leitura da tabela ja salva, se existir
#caso mudar as recompensas provavelmente precisa zerar a tabela e começar o treino de novo!!!
if os.path.exists('qTableSerializada.pickle'):
  with open('qTableSerializada.pickle', 'rb') as handle:
    qTable = pickle.load(handle)   

if os.path.exists('qTableContadorSerializada.pickle'):
  with open('qTableContadorSerializada.pickle', 'rb') as handle:
    qTableContador = pickle.load(handle)  

def dec2bin(dec):
    binN = []
    while dec != 0:
        binN.append(dec % 2)
        dec = dec / 2
    return binN

def jogando(ram):
    if ram[0x0100] != 14:
        return 1
    else:
        return False
    
def morreu(ram):
    if ram[0x0071] == 9:
       return 1
    else:
       return 0
    
def cutscene(ram):
    if ram[0x0071] != 00:
       return 1
    else:
       return 0

def estadoMudou(estado_anterior, estado_futuro):
    return not np.array_equal(estado_futuro, estado_anterior)

def printState(state):
  state_n = np.reshape(state.split(','), (2*radius + 1, 2*radius + 1))  
  #_=os.system("clear") #derruba desempenho
  #_=os.system("cls") #derruba desempenho
  mm = {'0':'  ', '1':'$$', '-1':'@@'}
  for i,l in enumerate(state_n):
    line = list(map(lambda x: mm[x], l))
    if i == radius + 1:
      line[radius] = 'XX'
    print(line) 

def getRam(env):
    ram = []
    for k, v in env.data.memory.blocks.items():
        ram += list(v)
    return np.array(ram)
    
def estadoSimp(vetorEstado):
  '''
  simplifica o espaço de estados pra facilitar o treino.
  O mario esta em [radius + 1][radius + 1], entao vamos considerar 4 blocos pra cima e 4 pra baixo e distanciaMax para frente, contando o mario (7*distanciaMax)
  '''
  vetorEstado = np.reshape(vetorEstado, ((2*radius) + 1, (2*radius) + 1))
  distanciaBloco    = np.zeros(8)
  distanciaInimigo  = np.zeros(8)
  distanciaMax      = 5
  indiceMaxSuperior = radius + 1 - 4
  indiceMaxInferior = radius + 1 + 4
  
  for j in range(indiceMaxSuperior, indiceMaxInferior): #3 pra cima e 3 pra baixo
    distanciaBloco[j - indiceMaxSuperior]   = distanciaMax
    distanciaInimigo[j - indiceMaxSuperior] = distanciaMax
    for i in range(radius, radius + distanciaMax): #até distanciaMax pra frente
      if vetorEstado[j][i] == 1:    # ve se tem bloco pra pisar
        distanciaBloco[j - indiceMaxSuperior] = i - radius
        break
      elif vetorEstado[j][i] == -1: # ve se tem inimigo
        distanciaInimigo[j - indiceMaxSuperior] = i - radius
        break

  estadoSimplificado = np.append(distanciaBloco, distanciaInimigo)
      
  print(vetorEstado)
  print(distanciaBloco)
  print(distanciaInimigo)
  print("vetor estado simplificado:")
  print(estadoSimplificado)
  return estadoSimplificado

def politica(estado_atual):
  #balanceamento entre exploração e explotação
  e = 0.8
  if np.random.rand() > e:
    return indiceMaxQ(str(estado_atual))
  else:
    return np.random.randint(0,qtdAcoes)

def politicaOtima(estado_atual):
  print(qTable[str(estado_atual)])
  return indiceMaxQ(str(estado_atual))

#ver necessidade
def recompensaAcao(acao):
  if acao in [128,129,130,131]:
     return 4
  else:
     return 0

def valorQ(estado_atual, acao):
  if str(estado_atual) in qTable:
    return qTable[str(estado_atual)][acao]
  else:
    qTable[str(estado_atual)] = np.zeros(qtdAcoes)
    #heuristica estados desconhecidos: pra dar mais qualidade em correr pra correr pra direita se nao tiver entrada na qTable
    qTable[str(estado_atual)][4] = .8    #qualidade em pular pra direita
    qTable[str(estado_atual)][5] = 1   #qualidade em correr pra direita
    qTable[str(estado_atual)][2] = -.2  #qualidade em andar pra esquerda
    qTable[str(estado_atual)][2] = .7  #qualidade em pular
    return qTable[str(estado_atual)][acao]

def indiceMaxQ(estado_atual):
  max = 0
  indice = 0
  for i in range(qtdAcoes):
    Q = valorQ(estado_atual, i)
    if Q > max:
      max = Q
      indice = i
  return indice

def contadorEstado(estado):
  stringEstado = str(estado)
  if stringEstado in qTableContador:
    qTableContador[str(estado)] = qTableContador[str(estado)] + 1
    return qTableContador[str(estado)]
  else:
    qTableContador[str(estado)] = 1
    return 1

def atualizaQ(estado, acao, estadoFuturo, recompensa):
  #alfa = 0.8
  cont = contadorEstado(estado)
  print("contador desse estado=: " + str(cont))
  alfa = 500/(500 + cont)
  gama = .6
  Q = valorQ(estado, acao) + alfa * (recompensa + gama*valorQ(estadoFuturo,indiceMaxQ(estadoFuturo)) - valorQ(estado, acao))
  #Se tiver no dicionario atualiza q, se o estado nao tiver inicializa a entrada
  if str(estado) in qTable:
    qTable[str(estado)][acao] = Q
  return   

def train():
  env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
  estado = 0
  utilidade = 0
  t = 0
  conta = 0
  while conta==0:
    env.reset()

    
    acao = 0
    x = 0
    quadroAcao = 0
    estado = np.zeros(121)
    
    #loop roda até acabar o tempo ou até morrer
    for quadros in range(6000):  
      ram = getRam(env)
      terminou  = not jogando(ram)
      vivo      = not morreu(ram)

      env.render()
      
      estado_novo, x_novo, y = getInputs(ram, radius)
      print("x = " + str(x) +"x_novo = " + str(x_novo))
      if estadoMudou(estado_novo, estado):           
        recompensa = -1 + (x_novo - x)*10 - (not vivo)*10000
        if terminou:
           recompensa = recompensa + 10000
        print("recompensa novo estado: " + str(recompensa))
        atualizaQ(estadoSimp(estado), acao, estadoSimp(estado_novo), recompensa)
        #print("tabela Q atualizada:")
        #print(qTable[str(estadoSimp(estado_novo))])
        estado = estado_novo 
        x = x_novo
        state_mtx = np.reshape(estado, ((2*radius) + 1, (2*radius) + 1))
        printState(getState(ram, radius)[0])

      #toma uma açao a cada 3 quadros
      if quadros - quadroAcao > 3:
        quadroAcao = quadros
        acao = politica(estadoSimp(estado_novo))

      #age em cima do estado
      ob, rew, done, info = env.step(dec2bin(actions_list[acao]))
      if terminou or not vivo:
        with open('qTableSerializada.pickle', 'wb') as handle:
          pickle.dump(qTable, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('qTableContadorSerializada.pickle', 'wb') as handle:
          pickle.dump(qTableContador, handle, protocol=pickle.HIGHEST_PROTOCOL)
        break            
          
    #Determina a utilidade (recompensa final) da jogada e considera as moedas, mas acho que isso pode confundir o treino porque ele nao ve moeda no jogo 
    utilidade = 2*x - 0.1*quadros - (not vivo)*100000 + terminou*10000
    t=t+1
    if(utilidade>0):
       print("Tentativa",t)
       print("Passou")
       conta+=1
  env.close()
  print(rew)
  return

def main():  
      train()

    
if __name__ == "__main__":
  main()  
