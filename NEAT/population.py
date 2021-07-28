import threading
import copy
#import multiprocessing as mp
class Pop:
    """
    size = 0
    global main
    global birds
    global obstacles
    global windObj
    global bird
    """
    
    def __init__(self, size, bird, main):
        self.size = size
        self.string = "bird"
        self.main = main
        self.bird = bird
        self.birds = [copy.deepcopy(bird) for i in range(self.size)]
        self.threads = []
        self.procs =[]


    def run(self, interval):
        self.birds = [copy.deepcopy(self.bird) for i in range(self.size)]
        self.threads.clear()
        self.procs.clear()
        num = interval
        #print("hey", num)
        for i in range(len(self.birds)):
            #global vars()[string+str(i)]
            #globals()[self.string+str(i)] = threading.Thread(target = self.main, args=(self.birds[i], copy.deepcopy(num),))
            #print(str(vars()[self.string+str(i)]))
            self.threads.append(threading.Thread(target = self.main, args=(self.birds[i], copy.deepcopy(num),)))
            self.threads[i].start()
            #self.procs.append(mp.Process(target = self.main, args=(self.birds[i], copy.deepcopy(num),)))
            #self.procs[i].start()
            #(globals()[self.string+str(i)]).start()
            #(vars()[self.string+str(i)]).join()
            num+=1
        checking = threading.Thread(target = self.check)
        #checking = mp.Process(target=self.check)
        checking.start()

    def check(self):
        while (not len(self.birds)==0):
            #print(bird0.isAlive())
            for i, bird in enumerate(self.birds):
                if bird.isDead() and (len(self.birds) !=0):
                    try:
                        #print(str(i) + ", " + str(len(self.birds)))
                        del self.birds[i]
                        #self.threads = [t for t in self.threads if (t != self.threads[i])]
                        del self.threads[i]
                        #procs[i].terminate()
                        #procs[i].join()
                        #del procs[i]
                    except Exception as e:
                        print(e)
                        #pass
                    #print(len(self.birds))
        #self.run()
        #print("done")
