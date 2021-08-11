
# NEAT  

## About  

NEAT (NeuroEvolution of Augmenting Topologies) is a genetic algorithm used to evolve artificial neural networks. More can be read from the [original paper]([http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)) by Kenneth O. Stanley and Risto Miikkulainen. The algorithm itself will not be discussed here. This repository is an implementation of NEAT.  A couple of examples are included here as well.  
  

## Implementation   

In this implementation, three classes were created: Genes, Population, NEAT.  
  

**Genes**  
This class was created to handle all gene related data and methods used in the algorithm.   
A base Genome in this implementation is represented as follows:  
`[[[Input_Node_1, Bias_1], ....[Input_Node_m, Bias_m], [Output_Node_1, Bias_m+1],..[Output_Node_n, Bias_m+n]],  
[[Input_Node_1, Output_Node_1, Innovation_Num_1, weight_1, Connection_Status (True/False)],....[Input_Node_m, Output_Node_n, Innovation_Num_m*n, weight_m*n, Connection_Status]]’`  
As time progresses, base genomes evolve into more complicated genomes which usually do a better job at the given task. The program trying to implement NEAT need not directly use any of the methods or data from this class as they are handled by the NEAT class.  
  

**Population**  
This class handles the population of each generation or iteration at runtime. In other words, birds (Bots, organisms, Agents that will be performing the task) will be deleted/killed or spawned on certain conditions such as repopulation or failure of the given task.   
  

**NEAT**  
This is the actual class which will be used to implement NEAT in the program. 

## Implementation
The program implement this implementation of NEAT in the following way:
```python
import  neat2  as  neat
import  threading
....

pop_size = ... //The population size of one generation
iteration_num = ... //The number of agents to be run at a time. This value is less than or equal to pop_num
gen_num = ...
num_inputs = ... //The number of inputs to the Neural Networks
num_outputs = ... //The number of outputs of the Neural Networks
cutoff = ... //The top (best) fraction of the organisms to take part in reproduction 

....

class Agent():
   def __init__(self):
      self.dead = False
      self.fitness = 1
      
   def isDead(self):
      return self.dead
   
   def die(self):
      self.dead = True


def func(agent, num):
   //This function is used for running the bird (agent) in the game/environment.
   
   while (not agent.isDead()):
      //render game/environment and agent (if possible)

      inputs = ... //inputs to the neural network. Usually what the agent has observed or the environment's states
      
      action = neat.decide(inputs, num) //These are the outputs of the neural netwrok. One can also perform an extra function on these outputs. 
      // An if-statement can also be used to decide an action

      agent.fitness  =  ... //update bird's fitness

      state = ... //state of the environment or agent's postion
      if (state == condition):
         bird.die()



def check():
 while count/pop_size <= gen_num:
    if (len(neat.getOrganisms()) == 0):
     // reset environment/game if required
     neat.runOrganisms() //repopulate the environment
       
 ....      
      
neat  =  neat.NEAT()

agent = Agent()

neat.createPopulation(pop_size, iteration_num, agent, func, num_inputs, num_outputs, cutoff)

neat.runOrganisms()

keep_repop = threading.Thread(target = check)

keep_repop.start()

```

***Note:***
`env.render()` from OpenAI's gym module does not support multithreading. The pendulum example included within this repository provides an alternative way to use `env.render()` and NEAT together. However, rendering an environment is not needed in running a bird in a gym environment.
