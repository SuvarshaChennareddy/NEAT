import population as popu
import Genes as genes
import numpy as np
from numpy.random import choice
import copy

class NEAT:
    def __init__(self):
        self.fitness = []
        self.gen = 1
        self.size = 1
        
    
    def createPopulation(self, popsize, oat, bird, main, numInputs, numOutputs, auto):
        self.organisms = genes.Genes(numInputs, numOutputs, popsize)
        self.organisms.newGenomes()
        self.pop = popu.Pop(oat, bird, main)
        self.fitness = [1]*popsize
        self.size = popsize
        self.inter = oat
        self.r = 0
        self.best = [0,0,0]
        self.cutoff = int(popsize*auto)
        

    def runOrganisms(self):
        if (self.r >= self.size):
            self.r = 0
            self.reproduce()
        else:
            self.pop.run(self.r)
            self.r += self.inter
        
    def getOrganisms(self):
        return self.pop.birds
    
    def setFitness(self, num, fitness):
        self.fitness[num] = fitness

    def getBest(self):
        return self.best

    def decide(self, ii, num, creature=None):
        if (creature is None):
            c = self.organisms.organs[num]
        else:
            c = creature
        result = self.organisms.eval(ii, c)
        return result
        

    def reproduce(self):
        self.fitness = self.organisms.speciate(self.organisms.organs,
                                                self.fitness,
                                                self.gen)


        species = copy.deepcopy(self.organisms.species)
        species.reverse()
        organisms = copy.deepcopy(self.organisms.organs)
        organisms.reverse()
        fitnesses = copy.deepcopy(self.fitness)
        fitnesses.reverse()
        if(fitnesses[len(fitnesses)-1] > self.best[0]):
            self.best=[fitnesses[len(fitnesses)-1], species[len(fitnesses)-1],
                   organisms[len(fitnesses)-1]]
        cgs = []
        for j in range(self.cutoff):
            cgs.append(organisms[len(fitnesses)-(j+1)])
        

        for i in range(len(self.organisms.representatives)):
            specnum = i+1
            if specnum in species:
                num = round((species.count(specnum))/4)
                g = [i for i, e in enumerate(copy.deepcopy(species)) if e == specnum]
                for j in range(num):
                    del species[g[num]]
                    del organisms[g[num]]
                    del fitnesses[g[num]]

        self.organisms.organs.clear()
        self.organisms.species.clear()
        self.fitness.clear()
        organisms.reverse()
        species.reverse()
        fitnesses.reverse()
        self.organisms.organs = copy.deepcopy(organisms)
        self.organisms.species = copy.deepcopy(species)
        self.fitness = copy.deepcopy(fitnesses)
                    
                
                
                
                       
        for i in range(len(self.organisms.organs)):
            self.fitness[i] = self.organisms.shared_fitness(self.organisms.organs[i],
                                                            self.fitness[i],
                                                            self.organisms.organs,
                                                            self.organisms.species)


            


        s = sum(self.fitness)
        gs = copy.deepcopy(self.organisms.organs)
        k= self.cutoff
        while k < self.size:
            intercross = np.random.uniform(0, 100)
            index1 = 0
            index2 = 0
            if (intercross <= 10):
                draw = choice(a = np.arange(len(self.organisms.organs)), replace=True,
                          size = 2,
                  p=[self.fitness[i]/s for i in range(len(self.fitness))])
                
                index1 = gs[draw.tolist()[0]]
                index2 = gs[draw.tolist()[1]]
            else:
                draw = choice(a = np.arange(len(self.organisms.organs)), size = 1, p = [self.fitness[i]/s for i in range(len(self.fitness))])
                index1 = gs[copy.deepcopy(draw.tolist()[0])]
                rindex1 = copy.deepcopy(draw.tolist()[0])
                if (self.organisms.species.count(self.organisms.species[(rindex1)]) == 1):
                    index2 = copy.deepcopy(index1)
                    rindex2 = copy.deepcopy(rindex1)
                else:   
                    while True:
                        draw = choice(a = np.arange(len(self.organisms.organs)), size = 1, p = [self.fitness[i]/s for i in range(len(self.fitness))])
                        rindex2 = copy.deepcopy(draw.tolist()[0])
                            
                        if (bool(self.organisms.species[(rindex1)] == self.organisms.species[rindex2])) and (bool(not(rindex2 == rindex1))):
                            break
                    index2 = gs[copy.deepcopy(draw.tolist()[0])]

            c = self.organisms.crossover(index1, self.fitness[self.organisms.organs.index(index1)],
                                     index2, self.fitness[self.organisms.organs.index(index2)])
            if (c in cgs):
                continue
            else:
                cgs.append(copy.deepcopy(c))
                k+=1
        self.organisms.organs.clear()
        self.organisms.organs = copy.deepcopy(cgs)
        del cgs[:]
        self.fitness.clear()
        self.fitness = [1]*self.size
        self.gen+=1
        self.organisms.create_nns()
        
        self.runOrganisms()
