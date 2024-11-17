#!/usr/bin/env python
# marioRule.py
# Author: Fabrício Olivetti de França
#
# Agente baseado em regras para o SMW

import sys
import retro
import pickle


import numpy as np
from numpy.random import uniform, choice, random
import pickle
import os
import time
from rominfo import *
from train import *


qTable = {} #dicionario usa hashtable, mais rapida a busca dos estados

#faz a leitura da tabela ja salva, se existir
#caso mudar as recompensas provavelmente precisa zerar a tabela e começar o treino de novo!!!
if os.path.exists('qTableSerializada.pickle'):
  with open('qTableSerializada.pickle', 'rb') as handle:
    qTable = pickle.load(handle)   
if os.path.exists('qTableContadorSerializada.pickle'):
  with open('qTableContadorSerializada.pickle', 'rb') as handle:
    qTableContador = pickle.load(handle) 

def politicaOtima(estado_atual):
  #precisa usar um termo bem pequeno de exploração pra case de cutscene ele conseguir destravar
  e = 0.95
  if np.random.rand() > e:
    return indiceMaxQ(str(estado_atual))
  else:
    return np.random.randint(0,qtdAcoes)

def play():
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
    estado = 0

    env.reset()
    env.step(dec2bin(actions_list[1]))

    utilidade = 0
    acao = 0
    quadroAcao = 0

    #loop roda até acabar o tempo ou até morrer
    for quadros in range(20000):  
        ram = getRam(env)
        terminou  = not jogando(ram)
        vivo      = not morreu(ram)

        env.render()
        
        #faz uma acao a cada mudança de estado
        estado, x, y = getInputs(ram, radius)

        #toma uma açao a cada 5 quadros
        if quadros - quadroAcao > 3:
            quadroAcao = quadros
            acao = politicaOtima(estadoSimp(estado))

        #age em cima do estado
        ob, rew, done, info = env.step(dec2bin(actions_list[acao]))
        print(info)
        print(quadros)
        if terminou or not vivo:
            break            
        
    #Determina a utilidade (recompensa final) da jogada e considera as moedas, mas acho que isso pode confundir o treino porque ele nao ve moeda no jogo 
    utilidade = 2*x - 0.1*quadros - (not vivo)*100000 + terminou*10000 + info["score"]
    print("utilidade = " + str(utilidade))

    env.close()
    return

def main():  
    play()

    
if __name__ == "__main__":
  main()  
