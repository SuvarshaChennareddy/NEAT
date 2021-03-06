import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")
class Genes:
    def __init__(self, innum, outnum, num):
        self.innum = innum
        self.outnum = outnum
        self.size = innum * outnum
        self.num = num
        self.kk = []
        global species
        self.species = []
        self.representatives = []
        self.organs = []
        self.mutinnov = []
        self.mutnode = []
        global innov
        self.innov = []
        self.biggestnode = self.innum + self.outnum
        self.nns = []
    
    def create_nns(self):
        self.nns.clear()
        for genome in self.organs:
            node_orderB = genome[0]
            node_cons = genome[1]
            node_order = [node_orderB[i][0] for i in range(len(node_orderB))]
            hidden_nodes = node_order[self.innum:]
            node_bias = [node_orderB[i][1] for i in range(len(node_orderB))]
            nn = np.zeros((len(node_order)+1, len(node_order)))
            nn = nn.tolist()
            for i in range(len(node_order)):
                nn[len(node_order)][i] = [node_bias[i], 0, True]
            for i in range(len(genome[1])):
                if node_cons[i][4]:
                    nn[node_order.index(node_cons[i][0])][node_order.index(node_cons[i][1])] = [0, node_cons[i][3], node_cons[i][4]]
            self.nns.append(nn)
    def newGenomes(self):
        self.gene1=[(copy.deepcopy([i+1,1])) for i in range(self.innum+self.outnum)]#np.arange(1, self.innum+self.outnum+1)
        for j in range(self.num):
            self.kk = []
            self.a = 0
            self.k = 1
            for i in range(self.size):
                self.a+=1
                self.conn = [self.a]
                self.conn.append(self.innum+self.k)
                if (self.a >= self.innum):
                    self.a = 0
                    self.k+=1
                innovation = self.innovnum(copy.deepcopy(self.conn))
                self.conn.append(innovation)
                self.conn.append(np.random.uniform(-5,5))
                self.conn.append(True)
                self.kk.append(self.conn)
            vars()["genome"+str(j)] = [self.gene1, self.kk]
            self.organs.append(vars()["genome"+str(j)])
        self.create_nns()
    
    def innovnum(self, con):
        if con in self.innov:
            return self.innov.index(con)
        else:
            self.innov.append(con)
            return self.innovnum(con)
    
        
    def mutinnovnum(self, con):
        if con in self.mutinnov:
            return self.mutnode[self.mutinnov.index(con)]
        else:
            self.mutinnov.append(con)
            self.biggestnode+=1
            self.mutnode.append(copy.deepcopy(self.biggestnode))
            return self.mutinnovnum(con)
    def eval(self, inputs, genome):
        inputs = inputs
        node_orderB = genome[0]
        node_cons = genome[1]
        node_order = [node_orderB[i][0] for i in range(len(node_orderB))]
        hidden_nodes = node_order[self.innum:]
        if (genome in self.organs):
            nn = self.nns[self.organs.index(genome)]
        else:
            node_bias = [node_orderB[i][1] for i in range(len(node_orderB))]
            nn = np.zeros((len(node_order)+1, len(node_order)))
            nn = nn.tolist()
            for i in range(len(node_order)):
              nn[len(node_order)][i] = [node_bias[i], 0, True]
            for i in range(len(genome[1])):
              if node_cons[i][4]:
                nn[node_order.index(node_cons[i][0])][node_order.index(node_cons[i][1])] = [0, node_cons[i][3], node_cons[i][4]]
        
        for i in range(self.innum):
            nn[i][0] = [inputs[i], 0, True]
        total_value = []
        node_value = 0


        for x in range(len(hidden_nodes)):
            try:
                column = [nn[i][self.innum+x] for i in range(len(node_order)+1)]
            except:
                print(genome)
                print(nn)
            for y in range(len(node_order)+1):
                if ((not(y==len(node_order))) and (type(column[y]) == list)):
                    try:
                        column[y][0] = column[y][1]*nn[y][0][0]
                    except:
                        column[y][0] = column[y][1]*nn[y][0]
                        

                if (type(column[y]) == list):
                    node_value += column[y][0]
            nn[self.innum+x][0] = self.sigmoid(node_value)
        for i in range(self.outnum):
            total_value.append(nn[node_order.index(self.innum+1+i)][0])
        return total_value

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def crossover(self, genome1, fitness1, genome2, fitness2):
        child_genome = [[],[]]
        genome1 = copy.deepcopy(genome1)
        genome2 = copy.deepcopy(genome2)
        fitness2 = copy.deepcopy(fitness2)
        fitness1 = copy.deepcopy(fitness1)
        prob = (fitness1/(fitness1 + fitness2))*100
        innov_nums1 = [genome1[1][i][2] for i in range(len(genome1[1]))]
        innov_nums2 = [genome2[1][i][2] for i in range(len(genome2[1]))]
        innov_nums = list(set(innov_nums1).intersection(set(innov_nums2)))
        innov_nums5 = list(set(copy.deepcopy(innov_nums1)).union(set(copy.deepcopy(innov_nums2))))
        for y in enumerate(innov_nums5):
            num = np.random.uniform(0,101)
            if (num <= prob):
                try:
                    #genome2 is good but the given innovation number is not in it, then pass and dont do anything 
                    if ((fitness2 > fitness1) and (y[1] not in innov_nums2)):
                        pass
                    else:
                        child_genome[1].append(genome1[1][innov_nums1.index(y[1])])
                except:
                    if (y[1] in innov_nums2):
                        child_genome[1].append(genome2[1][innov_nums2.index(y[1])])
            else:
                try:
                    if ((fitness1 > fitness2) and (y[1] not in innov_nums1)):
                        pass
                    else:
                        child_genome[1].append(genome2[1][innov_nums2.index(y[1])])
                except:
                    if (y[1] in innov_nums1):
                        child_genome[1].append(genome1[1][innov_nums1.index(y[1])])
                        
        child_genome[0].extend(self.order_nodes(child_genome, genome1, genome2))
        mutprob = np.random.randint(0, 100)
        if (mutprob <= 5):
            child_genome = self.choose(child_genome)
        return child_genome

    def order_nodes(self, genome, pgenome1, pgenome2):
        order_node = []
        connects = genome[1]
        b3 = []
        first = []
        connects = [[connects[i][0], connects[i][1]] for i in range(len(connects))]
        p1 = [pgenome1[0][i][0] for i in range(len(pgenome1[0]))]
        p2 = [pgenome2[0][i][0] for i in range(len(pgenome2[0]))]

        b1 = [pgenome1[0][i][1] for i in range(len(pgenome1[0]))]
        b2 = [pgenome1[0][i][1] for i in range(len(pgenome1[0]))]
    
        every_node = copy.deepcopy(p1)
        if(len(p1) < len(p2)):
            every_node.clear()
            every_node = copy.deepcopy(p2)
        hidden1 = (copy.deepcopy([i for i in copy.deepcopy(p2) if i not in copy.deepcopy(p1)]))
        hidden2 = (copy.deepcopy([i for i in copy.deepcopy(p1) if i not in copy.deepcopy(p2)]))
        hidden1.reverse()
        hidden2.reverse()
        for i in range(len(hidden1)):
            try:
                if (hidden1[i] not in every_node):
                    every_node.insert((every_node.index(p2[(p2.index(hidden1[i]))+1])), hidden1[i])
            except:
                print((every_node.index(p2[(p2.index(hidden1[i]))+1])))
                print(p2[(p2.index(hidden1[i]))+1])
                print(hidden1)
                print(hidden2)
                print(p1)
                print(p2)
                print(every_node)
        for i in range(len(hidden2)):
            try:
                if (hidden2[i] not in every_node):
                    every_node.insert((every_node.index(p1[(p1.index(hidden2[i]))+1])), hidden2[i])
            except:
                print((every_node.index(p1[(p1.index(hidden2[i]))+1])))
                print(p1[(p1.index(hidden2[i]))+1])
                print(hidden1)
                print(hidden2)
                print(p1)
                print(p2)
                print(every_node)
        for i in range(len(connects)):
            first.extend(connects[i])
        for i in range(len(every_node)):
            if (every_node[i] not in first):
                every_node[i] = None
        every_node = [x for x in every_node if x != None]
        for i in range(len(first)):
            if (first[i] not in every_node):
                print(hidden1)
                print(hidden2)
                print(every_node)
                print(genome)
                print(pgenome1)
                print(p1)
                print(pgenome2)
                print(p2)
        for i in range(len(every_node)):
            prob = np.random.uniform(0,101)
            if prob <=50:
                try:
                    b3.append(b1[i])
                except:
                    try:
                        b3.append(b2[i])
                    except:
                        b3.append(1)
                    #pass
            elif prob>=50:
                try:
                    b3.append(b2[i])
                except:
                    try:
                        b3.append(b1[i])
                    except:
                        b3.append(1)

        all_nodes = [[every_node[i], b3[i]] for i in range(len(every_node))]
        return all_nodes

    def mutation_weight(self,genome):
        childg = copy.deepcopy(genome)
        pro = np.random.uniform(0,101)
        if pro <=50:
            ran = np.random.randint(0, len(childg[1]))
            if (pro > 40):
                weight = np.random.uniform(-10,10)
            else:
                weight = childg[1][ran][3] + np.random.uniform(-1.5,1.5)
                                        
            childg[1][ran][3] = weight
            
        else:
            ran = np.random.randint(0, len(childg[0]))
            bias = np.random.uniform(-3,3)
            childg[0][ran][1] = bias

        return childg

    def mutation_add_node(self,genome):
        child_g = copy.deepcopy(genome)
        new_connection = []
        new_connection.clear()
        try:
            ran = np.random.randint(0, len(child_g[1]))
        except:
            print(child_g)
        end_node = copy.deepcopy(child_g[1][ran][1])
        starting_node = copy.deepcopy(child_g[1][ran][0])
        current_bnode = copy.deepcopy(self.mutinnovnum([starting_node, end_node])) #copy.deepcopy(self.biggestnode)
        child_nodes = [child_g[0][i][0] for i in range(len(child_g[0]))]
        if (current_bnode in child_nodes):
            return self.mutation_add_node(child_g)
        else:
                
            child_g[1][ran][1] = copy.deepcopy(current_bnode)
            old_wight = copy.deepcopy(child_g[1][ran][3])
            child_g[1][ran][3] = copy.deepcopy(1)
            innvo = copy.deepcopy(self.innovnum(copy.deepcopy([starting_node, current_bnode])))
            child_g[1][ran][2] = copy.deepcopy(innvo)
            new_connection.append(current_bnode)
            new_connection.append(end_node)
            innov_numb = copy.deepcopy(self.innovnum(copy.deepcopy([new_connection[0], new_connection[1]])))
            new_connection.append(innov_numb)
            new_connection.append(old_wight)
            new_connection.append(True)
            child_g[1].append(copy.deepcopy(new_connection))
            child_nodes = [child_g[0][i][0] for i in range(len(child_g[0]))]
            try:
                index_old1 = child_nodes.index(end_node)
            except:
                print(child_g)
                print(child_nodes)
                print(end_node)
            child_g[0].insert(index_old1, copy.deepcopy([current_bnode, 1]))
            return child_g

    def mutation_add_connection(self,genome):
        chilg = copy.deepcopy(genome)
        cou = 0
        try:
            new_conne = []
            connectis = chilg[1]
            connectis = copy.deepcopy([[connectis[i][0], connectis[i][1]] for i in range(len(connectis))])
            child_nods = [chilg[0][i][0] for i in range(len(chilg[0]))]
            while True:
                cou +=1
                startin = child_nods[:-self.outnum]
                ran1 = np.random.randint(0, len(startin))
                ran3 =  copy.deepcopy(ran1)
                stnode = startin[ran1]
                endin = copy.deepcopy(child_nods)
                if stnode <= self.innum:
                    ran3 = self.innum+1
                ran2 = np.random.randint(ran3, len(endin))
                ennode = endin[ran2]
                if (([stnode, ennode] not in connectis)):
                    break
                if (cou == 10):
                    return childg
                    
            new_conne.append(stnode)
            new_conne.append(ennode)
            innov_numbb = copy.deepcopy(self.innovnum(copy.deepcopy([new_conne[0], new_conne[1]])))
            new_conne.append(innov_numbb)
            new_conne.append(np.random.uniform(-10,10))
            new_conne.append(True)
            chilg[1].append(new_conne)
            return chilg
        except:
            return chilg

        
        
        

    def mutation_enable_disable(self,genome):
        childgg = copy.deepcopy(genome)
        rano = np.random.randint(0, len(childgg[1]))
        bool_value = copy.deepcopy(childgg[1][rano][4])
        childgg[1][rano][4] = not bool_value
        return childgg
        
    def choose(self, child_genome):
        choice = np.random.choice(a=[1,2,3,4], replace = True, size=1, p=[0.01, 0.95, 0.02, 0.02])
        choice = choice.tolist()[0]
        if choice == 1:
            return self.mutation_add_node(child_genome)
        elif choice == 2:
            return self.mutation_weight(child_genome)
        elif choice == 3:
            return self.mutation_enable_disable(child_genome)
        else:
            return self.mutation_add_connection(child_genome)

    def speciate(self, genomes, f_scores, gen):
        self.species.clear()
        genomess = copy.deepcopy(genomes)
        fitnesses = copy.deepcopy(f_scores)
        sorted_genomes = [x for _,x in sorted(zip(fitnesses,genomess))]
        fitnesses.sort(reverse=True)
        sorted_genomes.reverse()
        self.organs.clear()
        self.organs = copy.deepcopy(sorted_genomes)        
        for i in range(len(sorted_genomes)):
            if (gen == 1) and (i == 0):
                self.representatives.append(sorted_genomes[0])
                self.species.append(copy.deepcopy(len(self.representatives)))
            else:
                for j in range(len(self.representatives)):
                    checker = copy.deepcopy(self.representatives[j])
                    try:
                        dis = copy.deepcopy(self.distance(1, 1, 1, None, copy.deepcopy(sorted_genomes[i]), checker))
                    except:
                        print(copy.deepcopy(sorted_genomes[i]))
                    if dis <= 2:
                        self.species.append(j+1)
                        break
                    
                if not(len(self.species) == i+1):
                    self.representatives.append(copy.deepcopy(sorted_genomes[i]))
                    self.species.append(copy.deepcopy(len(self.representatives)))
        return fitnesses



    def distance(self, c1, c2, c3, n, genome1, genome2):
        num_of_excess = 0
        num_disjoint = 0
        wight_avg = 0
        weight_diffs = []
        genome1 = copy.deepcopy(genome1[1])
        genome2 = copy.deepcopy(genome2[1])
        connects1 = [[genome1[i][3], genome1[i][2]] for i in range(len(genome1))]
        connects2 = [[genome2[i][3], genome2[i][2]] for i in range(len(genome2))]
        innov_nums1 = [genome1[i][2] for i in range(len(genome1))]
        innov_nums2 = [genome2[i][2] for i in range(len(genome2))]
        innov_nums3 = list(set(innov_nums1).difference(set(innov_nums2)))
        innov_nums4 = list(set(innov_nums2).difference(set(innov_nums1)))
        innov_nums5 = list(set(innov_nums2).intersection(set(innov_nums1)))
        try:
            max_inov = max(innov_nums1)
        except:
            print(genome1)
            print(innov_nums1)
        max_inovv = max(innov_nums2)
        sorted_in1 = innov_nums1
        sorted_in2 = innov_nums2

        
        connects1 = [connects1[i][0] for i in range(len(connects1))]
        connects1 = [x for _,x in sorted(zip(sorted_in1,connects1))]
        connects2 = [connects2[i][0] for i in range(len(connects2))]
        connects2 = [x for _,x in sorted(zip(sorted_in2,connects2))]
        sorted_in1 = sorted(innov_nums1)
        sorted_in2 = sorted(innov_nums2)

        l = list(set(sorted_in1).union(set(sorted_in2)))
        l.sort()
        
        if (max_inov > max_inovv):
                in_smax = l.index(max_inovv)
                num_of_excess = len(l)-1 - in_smax
                
        elif (max_inov < max_inovv):
                in_smax = l.index(max_inov)
                num_of_excess = len(l)-1 - in_smax
            
        num_disjoint = len(innov_nums3)+len(innov_nums4) - num_of_excess


        for i in innov_nums5:
            try:
                diff = abs(connects1[sorted_in1.index(i)] - connects2[sorted_in2.index(i)])
            except:
                print(i)
            weight_diffs.append(copy.deepcopy(diff))
        try:
            weight_avg = (sum(weight_diffs) / len(weight_diffs))
        except:
            weight_avg = 0

        if (n is None):
            n = max(len(connects1), len(connects2))
                                               
            """
            if ((max(len(connects1), len(connects2))) < 20):
                n = 1
            """
                
        distance = (c1 * num_of_excess/n) + (c2 * num_disjoint/n) + (c3 * weight_avg)
        return distance
            
                
            
    def shared_fitness(self, genome, fitness, genomes, species):
        specnum = species[genomes.index(genome)]
        members = species.count(specnum)
        new_fit = fitness/members
        return new_fit      
        
