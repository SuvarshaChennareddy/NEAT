import threading
import copy
class Pop:

    def __init__(self, size, bird, main):
        self.size = size
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
            self.threads.append(threading.Thread(target = self.main, args=(self.birds[i], copy.deepcopy(num),)))
            self.threads[i].start()
            num+=1
        checking = threading.Thread(target = self.check)
        checking.start()

    def check(self):
        while (not len(self.birds)==0):
            for i, bird in enumerate(self.birds):
                if bird.isDead() and (len(self.birds) !=0):
                    try:
                        del self.birds[i]
                        del self.threads[i]
                    except Exception as e:
                        print(e)
