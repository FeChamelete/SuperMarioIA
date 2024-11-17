#!/usr/bin/env python
# marioAstar.py
# Author: Fabrício Olivetti de França
#
# A* algorithm for Super Mario World
# using RLE

import sys
import os
import pickle
import retro

from rominfo import *
from utils import *

sys.setrecursionlimit(10000)

# quais movimentos estarão disponíveis
moves = {'direita':128, 'corre':130, 'pula':131, 'spin':386, 'esquerda':64}

# raio de visão (quadriculado raio x raio em torno do Mario)
raio = 6

# Se devemos mostrar a tela do jogo (+ lento) ou não (+ rápido)
mostrar = True

# Classe da árvore de jogos para o Super Mario World
class Tree:
    def __init__(self, estado, filhos=None, pai=None, g=0, h=0, terminal=False, obj=False):
        self.estado   = estado
        self.filhos   = filhos # lista de filhos desse nó
        
        self.g = g
        self.h = h
        
        self.eh_terminal = terminal
        self.eh_obj      = obj
        
        self.pai = pai # apontador para o pai, útil para fazer o backtracking

    def __str__(self):
        return self.estado
  
# Encontra o melhor filho de tree
def melhor_filho(tree):

    # Se for um nó de game over, retorna nada
    if tree.eh_terminal:
        return None

    # Se não tem filhos, esse é o melhor filho do ramo
    if tree.filhos is None:
        return tree, tree.g + tree.h

    # Lista dos melhores filhos de cada ramo de tree
    melhor = [melhor_filho(tree.filhos[k]) 
                   for k, v in moves.items()]

    # Elimina os terminais
    naoTerminal = lambda x: (x is not None) and (not x[0].eh_terminal)
    melhor = [x for x in melhor if naoTerminal(x)]
   
    # Se não tem nenhum filho não-terminal
    # marca esse nó como terminal para 
    # não ter que percorrer novamente 
    if len(melhor) == 0:
        tree.eh_terminal = True
        return None
        
    # Recupera o melhor filho e retorna
    filho, score = sorted(melhor, key=lambda t: t[1])[0]

    return filho, score

# Nossa heurística é a quantidade
# de passos mínimos estimados para
# chegar ao final da fase
def heuristica(estado, x):
#    return (4800 - x)/8
    estNum = np.reshape(list(map(int, estado.split(','))), (2*raio+1,2*raio+1))
    dist = np.abs(estNum[:raio+1,raio+2:raio+7]).sum()
    return ((4800 - x)/8) + 0.3*dist
 
# Verifica se chegamos ao final   
def checaObj(estado, x):
    return x>4800

# Verifica se um nó é uma folha 
def folha(tree):
    return tree.filhos is None

# Joga uma partida usando uma
# sequência de ações
def emula(acoes):

    #env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1', players=1)
    env.reset()

    while len(acoes)>0 and (not env.data.is_done()):
        a = acoes.pop(0)
        estado, xn, y = getState(getRam(env), raio)
        r, d = performAction(a, env)
        if d or env.data.is_done():
            print(d, env.data.is_done())
        if mostrar:
            env.render()

    estado, x, y = getState(getRam(env), raio)
    
    return estado, x, env.data.is_done()
    
# Expande a árvore utilizando a heurística
def expande(tree):
    
    acoes = []
   
    # Se a árvore já for um nó folha
    # não tem ações a serem feitas 
    if folha(tree):
        raiz  = tree
        filho = tree
    else:

        # Busca pelo melhor nó folha
        filho, score = melhor_filho(tree)     
        
        # Retorna para a raiz gravando
        # as ações
        raiz = filho
        while raiz.pai is not None:
            neto  = raiz
            raiz = raiz.pai
            for k, v in moves.items():
               if raiz.filhos[k] == neto:
                   acoes.append(v)
        acoes.reverse()
        print('ACOES:  (  ', len(acoes), ' ): ',  acoes)
        
    
    obj = False

    # Gera cada um dos filhos e verifica se atingiu objetivo
    filho.filhos = {}
    maxX         = 0
    for k, v in moves.items():
        estado, x, over = emula(acoes + [v])
        maxX            = max(x, maxX)
        obj             = obj or checaObj(estado, x)
        filho.filhos[k] = Tree(estado, g=filho.g + 1, h=heuristica(estado,x),
                                pai=filho, terminal=over, obj=obj)
    
    print(estado)
    print('FALTA: ', heuristica(estado, maxX))
        
    return raiz, obj

# Verifica se a árvore já atingiu o objetivo
def atingiuObj(tree):
    if tree.eh_terminal:
        return tree.eh_obj, []
    if tree.filhos is None:
        return False, []
    for k, v in moves.items():
        obj, acoes = atingiuObj(tree.filhos[k])
        if obj:
            acoes = [v] + acoes
            return obj, acoes
    return False, []

# Gera a árvore utilizando A*
def astar():
    
    global mostrar
 
    # Gera a árvore com o estado inicial do jogo 
    global env
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1', players=1)
        
    env.reset()
    estado, x, y = getState(getRam(env), raio)
  
    tree = Tree(estado, g=0, h=heuristica(estado,x))

    # Se já existe alguma árvore, carrega
    if os.path.exists('AstarTree.pkl'):
        tree = pickle.load(open('AstarTree.pkl', 'rb'))

    # Repete enquanto não atingir objetivo    
    obj, acoes  = atingiuObj(tree)

    while not obj:
        tree, obj = expande(tree)

        # Grava estado atual da árvore por segurança
        fw = open('AstarTree.pkl', 'wb')
        pickle.dump(tree, fw)
        fw.close()
        
    obj, acoes = atingiuObj(tree)
    mostrar = True
    emula(acoes)

    return tree
  
def main():  
  tree = astar()
    
if __name__ == "__main__":
  main()
