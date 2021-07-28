import pygame, sys, math

sys.path.append("../NEAT")

from random import randrange as randH

from pygame.locals import *

import threading

import os

import population as pop

import neat2 as neat

import copy

import gc

import math


# Start pygame
pygame.init()


neat = neat.NEAT()

# Set up resolution

windowObj = pygame.display.set_mode( ( 640, 480) )

fpsTimer = pygame.time.Clock()

maxFPS = 30


# Ground Elevation (pixels)

groundLevel = 400



# Global colors

birdColor = pygame.Color('#222222')

backgroundColor = pygame.Color('#abcdef')

groundColor = pygame.Color('#993333')

fontColor = pygame.Color('#FFFFFF')



#fontObj = pygame.font.SysFont('ActionIsShaded', 16)



# Class for pipe obstacles

class Pipes:



    height = 0

    width = 60

    gap = 115

    pos = 550

    replaced = False

    scored = False



    # Randomize pipe location

    def __init__(self):

        self.height = randH(175, groundLevel - 35)



    # Moves the pipes along the ground, checks if they're off the screen

    def move(self, movement):

        self.pos += movement

        if( self.pos + self.width < 0 ):

            return False #Return false if we moved off the screen

        return True



    # Handles drawing the pipes to the screen

    def draw(self, surface):

        pygame.draw.rect( surface, groundColor, (self.pos, self.height, self.width, groundLevel - self.height))

        pygame.draw.rect( surface, groundColor, (self.pos, 0, self.width, self.height - self.gap))



# Class for the player

class Bird:



    pos = (0,0)

    radius = 20
    velocity = 0
    gravity = 3
    dead = False
    


    score = 0

    highScore = 0



    def __init__(self, newPos):

        self.pos = newPos
        self.fitness = 0



    # Handles drawing the bird to the screen

    def draw(self, surface):

        intPos = ( int(math.floor(self.pos[0])), int(math.floor(self.pos[1])) )



        pygame.draw.circle(surface, birdColor, intPos, self.radius)



    # Attempt to move the bird, make sure we aren't hitting the ground


    def move(self, movement):

        posX, posY = self.pos

        movX, movY = movement

        if( (posY + movY + self.radius) < groundLevel ):

            self.pos = (posX + movX, posY + movY)

            return True #Return if we successfuly moved

        self.pos = (posX, groundLevel - self.radius)

        return False

    
    def getInputs(self, pipe):
        posX, posY = self.pos
        pipeDistance = (pipe.pos) - posX
        pipeHeight = pipe.height
        dSky = posY
        dGround = groundLevel - posY
        radius = self.radius
        gap = pipe.gap
        Dgap1 = posY - (pipe.gap/2 + pipe.height)
        Dgap2 = posY - (pipe.height)
        velocity = self.velocity
        gravity = self.gravity
        width = pipe.width
        #print(pipe.height)
        #return [pipeDistance, Dgap, dSky, dGround]
        return [pipeDistance, Dgap2, posY]

        
        




    # Test for collision with the given pipe

    def collision(self, pipe):

        posX, posY = self.pos

        collideWidth = ( pipe.pos < posX + self.radius and posX - self.radius < pipe.pos + pipe.width)

        collideTop = ( pipe.height - pipe.gap > posY - self.radius )

        collideBottom = ( posY + self.radius > pipe.height )

        if (collideWidth and ( collideTop or collideBottom)):

            return True

        return False

    def Click(self):

        self.velocity = -20

    def die(self):
        self.dead = True

    def isDead(self):
        if self.dead:
            return True
        else:
            return False




def resetGame():

    global pipes
    del pipes[:]

    pipes = [Pipes()]

    




def pause():

    while True:

        for event in pygame.event.get():

            if event.type == QUIT:

                pygame.quit()

                sys.exit()

            elif event.type == KEYDOWN:

                if ( event.key == K_ESCAPE):

                    return




def run(bird, num):
    
    global windowObj
    global pipes
    global highScore
    counter = 0
    pygame.event.get()
    while not bird.isDead():
        counter += (50*(bird.score+1))
        #print(counter)
        if (bird.pos[1] <= 0):
            bird.die()
        windowObj.fill(backgroundColor)


        # Check for event
        
        hs = 1000
        best = pipes[0]

        for i, pipe in enumerate(pipes):
            j = pipe.pos+pipe.width
            if (hs > j) and (j > bird.pos[0]):
                hs = pipe.pos+pipe.width
                best = pipe
        
        inputs = bird.getInputs(best)
        #print("result", num, neat.decide(inputs, num))
        try:
            if (round(neat.decide(inputs, num)[0]) == 1):
                bird.Click()
        except:
            print(num)
            print(len(neat.organisms.organs))
            if (round(neat.decide(inputs, num)[0]) == 1):
                bird.Click()

            



        # Add acceleration from gravity

        bird.velocity += bird.gravity

        bird.fitness = counter - math.sqrt(((best.pos - bird.pos[0])**2) + ((best.height+best.gap - bird.pos[1])**2))
##        if bird.score >= 1:
##            #bird.fitness = 50*counter - (best.pos - bird.pos[0])
##            bird.fitness +=1
##        else:
##            bird.fitness += 0.5


        #print(best.pos)



        neat.setFitness(num, bird.fitness)
        #print("fitness ", neat.fitness)





        if (not bird.move((0, bird.velocity))):

            bird.die()

            return

            #bird.velocity = 0
        

        for pipe in pipes:

            if (bird.collision(pipe)):

                bird.die()
                
                return
            
            if (not pipe.scored and pipe.pos + pipe.width < bird.pos[0] ):
                bird.score+=1

                #bird.fitness += 5

                #del pipes[pipes.index(pipe)]

            #pipe.scored = True

        #scoreSurface = fontObj.render( 'Score: ' + str(score) + ' High: ' + str(highScore), False, fontColor)

        #scoreRect = scoreSurface.get_rect()

        #scoreRect.topleft = (windowObj.get_height() / 2 , 10)

        #windowObj.blit(scoreSurface, scoreRect)

        #pygame.draw.rect(windowObj, groundColor, (0, groundLevel, windowObj.get_width(), windowObj.get_height()) )
    #print("gay")
        bird.draw(windowObj)
        
        #pygame.display.update()

        fpsTimer.tick(maxFPS)

    
    






bird = Bird((windowObj.get_width() / 4 , windowObj.get_height() / 2))
global pipes
pipes = [Pipes()]
neat.createPopulation(25, 25, bird , run, 3, 1, 0.2)
def Pip():
    global pipes
    global windowObj
    global birds

    while True:
        pygame.event.get()
        birds = neat.getOrganisms()
        
        windowObj.fill(backgroundColor)
        
        pygame.draw.rect(windowObj, groundColor, (0, groundLevel, windowObj.get_width(), windowObj.get_height()) )

        if (len(birds) == 0):
            
            resetGame()

            gc.collect()
            
            neat.runOrganisms()
            #print(neat.organisms.organs)
            #print(neat.organisms.species)
            
        for event in pygame.event.get():

            if event.type == QUIT:

                pygame.quit()

                sys.exit()


        for pipe in pipes:

            if (not pipe.replaced and pipe.pos < windowObj.get_width() / 2 ) :

                pipes[len(pipes):] = [Pipes()]

                pipe.replaced = True

            pipe.draw(windowObj)

            """"

            if ( not pipe.scored and pipe.pos + pipe.width < bird.pos[0] ):

                bird.score += 1

                pipe.scored = True
            """

            if( not pipe.move(-10)):

                del pipe


        #scoreSurface = fontObj.render( 'Score: ' + str(score) + ' High: ' + str(highScore), False, fontColor)

        #scoreRect = scoreSurface.get_rect()

        #scoreRect.topleft = (windowObj.get_height() / 2 , 10)

        #windowObj.blit(scoreSurface, scoreRect)
        for bird in birds:
            bird.draw(windowObj)


        pygame.display.update()
        
        fpsTimer.tick(maxFPS)




        # Draw stuff

pip = threading.Thread(target = Pip)
pip.start()
neat.runOrganisms()

