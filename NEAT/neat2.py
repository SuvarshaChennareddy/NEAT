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
        #self.fitness = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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

    def decide(self, ii, num):
        #creature =[[[1, 1], [2, 0.5799699133877518], [3, -0.8939348118547463], [4, 2.7685317621249617], [7, -1.7491854774977285], [6, 1], [5, 1]], [[1, 5, 0, 0.8949671954795315, True], [2, 5, 1, -8.017562700096855, True], [3, 5, 2, -2.754505331165879, True], [4, 5, 3, 2.3843768713631697, True], [3, 6, 4, 1, True], [6, 5, 5, -2.2870927069395264, True], [1, 7, 6, -1.382039230603106, True], [7, 5, 7, -3.6168660292519696, True], [7, 7, 8, -7.394334684569788, True], [2, 6, 9, 1.1918210550346853, True], [7, 6, 10, 1.2369825687776133, True]]]
        #[[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]], [[1, 5, 0, -1.0063231554931704, True], [2, 5, 1, -5.858837142358039, True], [3, 5, 2, -1.850369691944195, True], [4, 5, 3, 1.600983448880541, True]]]
        #[[[1, 2.7745551732446643], [2, -2.092758413173028], [3, 2.702279034096528], [4, 0.21170298632983897]], [[1, 4, 0, -0.9411976224638612, True], [2, 4, 1, 2.137762322375943, True], [3, 4, 2, 0.1915945151981946, True]]]
        #[[[1, 1], [2, 0.6187312768150011], [3, 0.5229912936135941], [5, 1], [4, 1]], [[1, 4, 0, -1.1860802219928732, True], [2, 4, 1, -5.812559146975895, True], [3, 4, 2, 1.0403346850675055, False], [1, 5, 3, 1.5299039173716786, True]]]
        #[[[1, 1], [2, 1], [3, 1], [9, 1], [7, 1], [12, 1], [10, 1], [4, 1]], [[1, 4, 0, 0.9059521407190809, True], [2, 4, 1, 2.7088626162370124, True], [3, 4, 2, 0.22934484395565313, True], [1, 7, 7, 1, True], [1, 9, 11, 1, True], [9, 7, 12, 1, True], [7, 10, 14, 1, True], [10, 4, 15, -1.6267334585666031, True], [7, 12, 20, 1, True]]]
        #[[[1, 1], [2, 1], [3, 1], [9, 1], [6, 1], [7, -0.1864532517464097], [4, 1]], [[1, 4, 0, -0.43534777018145254, False], [2, 4, 1, -7.291005254722474, True], [3, 4, 2, 0.06436533911627862, True], [1, 6, 5, 1, True], [6, 4, 6, -1.8794972834542345, True], [7, 4, 8, 1.9831436350033123, True], [6, 6, 11, 0.9908271627153287, True], [3, 7, 12, -7.661891749589671, True], [6, 9, 14, 1, True], [9, 6, 15, 0.9908271627153287, True]]]
        #[[[1, 1], [2, 1], [3, -2.76081897128282], [6, 1], [32, 1], [34, 1], [13, -2.326507044325547], [27, 1], [17, 1], [24, 1], [30, 1], [7, 1], [9, 1], [4, 1]], [[1, 4, 0, -3.1519368606729734, True], [2, 4, 1, -4.929984921673194, True], [3, 6, 6, 1, True], [6, 4, 7, 2.915005665920754, True], [6, 7, 8, 1, True], [7, 4, 9, 2.915005665920754, True], [2, 9, 12, 1, True], [9, 4, 13, -4.929984921673194, True], [3, 7, 18, 8.615505737243062, True], [6, 6, 21, 7.722482210749085, True], [7, 9, 23, -0.18178081748794916, True], [9, 9, 24, -3.2023185425679053, True], [3, 9, 25, 2.7247758438953333, True], [1, 7, 26, 7.4885656077553655, True], [2, 7, 27, -5.7555131008209255, False], [1, 13, 29, 1, False], [13, 7, 30, 7.4885656077553655, True], [7, 7, 31, -4.888527268714597, False], [1, 9, 34, -0.19707993231344467, True], [13, 13, 39, -2.9104234277407137, True], [6, 17, 40, 1, True], [17, 7, 41, 1, True], [17, 9, 56, -2.947010783181665, True], [17, 24, 59, 1, True], [24, 7, 60, 1, True], [1, 27, 67, 1, True], [27, 17, 68, -6.179015381817297, True], [3, 13, 69, 9.314003889686905, True], [24, 30, 75, 1, True], [30, 7, 76, 1, True], [3, 30, 82, -5.982998636200372, True], [3, 32, 84, 1, True], [32, 13, 85, 9.314003889686905, True], [1, 34, 88, 1, False], [34, 13, 89, 1, True]]]
        creature = self.organisms.organs[num]
        result = self.organisms.eval(ii, creature)
        return result
        

    def reproduce(self):
        self.fitness = self.organisms.speciate(self.organisms.organs,
                                                self.fitness,
                                                self.gen)
        #print(self.organisms.species)
        #print(self.fitness)

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
        #print(type(self.cutoff))
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
        #print(self.fitness)

            


        s = sum(self.fitness)
        #print(self.fitness)
        #print(s)
        #print(self.organisms.organs)
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
                #print("rindex1 ", rindex1)
                if (self.organisms.species.count(self.organisms.species[(rindex1)]) == 1):
                    index2 = copy.deepcopy(index1)
                    rindex2 = copy.deepcopy(rindex1)
                else:   
                    while True:
                        draw = choice(a = np.arange(len(self.organisms.organs)), size = 1, p = [self.fitness[i]/s for i in range(len(self.fitness))])
                        rindex2 = copy.deepcopy(draw.tolist()[0])
                            
                        #print("rindex2 ", rindex2)
                        if (bool(self.organisms.species[(rindex1)] == self.organisms.species[rindex2])) and (bool(not(rindex2 == rindex1))):
                            break
                    index2 = gs[copy.deepcopy(draw.tolist()[0])]
            #self.organisms.organs.index(index1)

            c = self.organisms.crossover(index1, self.fitness[self.organisms.organs.index(index1)],
                                     index2, self.fitness[self.organisms.organs.index(index2)])
            if (c in cgs):
                continue
            else:
                cgs.append(copy.deepcopy(c))
                k+=1
        self.organisms.organs.clear()
        self.organisms.organs = copy.deepcopy(cgs)
        #print(self.organisms.organs, len(self.organisms.organs))
        del cgs[:]
        self.fitness.clear()
        self.fitness = [1]*self.size
        self.gen+=1
        self.organisms.create_nns()
        

        #print(self.organisms.organs)
        #print(self.organisms.species)
        #print(self.organisms.representatives)
        #print()
        
        self.runOrganisms()
