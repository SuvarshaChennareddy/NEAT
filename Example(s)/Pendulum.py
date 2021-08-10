import sys
sys.path.append("../NEAT")
import neat2 as neat
import gym
import time
from math import sqrt, acos, sin
import threading
from threading import Lock
import copy
import numpy as np
 
global o
o = 50
global org
org=0
global agent
global num
gen = 200
class Agent():
    def __init__(self):
        self.dead = False
        self.fitness = 1
    def isDead(self):
        return self.dead
    def die(self):
        self.dead = True
def render(env,count):
    if(count%5==0):
        env.render()

def setAgentAndNum(worker, number):
    global agent
    global num
    agent = worker
    num = number
    
global NegMin
NegMin=0
main_lock = Lock()
neat = neat.NEAT()

def addNegMin(score):
    global NegMin
    return score-NegMin
    
global env
env = gym.make("Pendulum-v0")
def addPoints(ag, count, state, next_state, last_action, action, reward):
    cosa=next_state[0]
    sina=next_state[1]
    angvel=next_state[2]
    
    exvel = -8*sin(((sina/abs(sina))*(acos(cosa)))/2)
    rew = 1.5*cosa + reward - (1/8)*(abs(exvel-angvel))
    return rew
 
def run():
    global NegMin
    global env
    global o
    global org
    global gen
    while (not (org/o >= gen)):
        global agent
        global num
        new_state = env.reset()
        state = [0,0,0]
        done = False
        action = 0
        last_action = 0
        count = 0
        diff = 0
        avgvel =0
        tot_rew = 0
        
        while (not done):
            count+=1
            with main_lock:
                if(count%5==0):
                    env.render()
            
            new_state, reward, done, _ = env.step([last_action])
            tot_rew+=abs(reward)
            if(done):
                agent.die()
                break
            ac_reward = addPoints(agent, count, state, new_state, last_action, action, reward)
            inputs = list(copy.deepcopy(new_state))
            inputs.append(copy.deepcopy(reward))
            action= (neat.decide(inputs, num)[0]-0.5)*4
                
            agent.fitness += addPoints(agent, count, state, new_state, last_action, action, reward)
            diff += (abs(last_action-action))
            avgvel += 8-abs(new_state[2])
            state = new_state
            last_action = action
            
        if (agent.fitness < NegMin):
           NegMin = agent.fitness
        fitness = addNegMin(agent.fitness)+2
        neat.setFitness(num, 5*(fitness))
        print(fitness, num, (org/o)-((org/o)%1))
        org+=1
        neat.runOrganisms()
 
 
agent0 = Agent()
neat.createPopulation(o, 1, agent0, setAgentAndNum, 4, 1, 0.2)
neat.runOrganisms()
run()

