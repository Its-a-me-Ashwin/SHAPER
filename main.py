import pymunk
from random import sample
from Agent.Agent2 import *
#from Physics.arm2 import Arm1
#from Physics.polygon import Polygon
#from Physics.utils import *
from multiprocessing import Pool

FramesPerAgent = 60*120 ## 2 mins of real timee if rendered.
PHYSICS_FPS = 25

Population = 20
LowPassFilter = 0.8

ScoreForAngle = 0.8
ScoreForPosition = 0.75

## Top percentage of the population will be used as parents and 1-TopPic will be the children.
TopPic = 0.4
ParentsCount = int(TopPic*Population)
ChildrenCount = Population-ParentsCount

## Start the learning. 
def main():
    ## Agents is a list of tuples.
    ## Each touple has the agent and the score. Need to change the inputs and outputs.
    Agents = []
    eta = 0.4
    for _ in range(Population):
        lAgent = Agent()
        lAgent.addLayer("Input", 59, None, False)
        lAgent.addLayer("H0", 128, TanH, False)
        lAgent.addLayer("H1", 256, TanH, False)
        lAgent.addLayer("H2", 128, TanH, False)
        lAgent.addLayer("H3", 64, TanH, False)
        lAgent.addLayer("Output", 9, None, True)

        Agents.append([lAgent, 0.0])

    generation = 0
    while True:
        print("Generation:", generation)
        generation += 1
        ## Generate the polygon and the goal orientation. For now this is constant.
        ## Goal consists of the final poistion and angle of the body and the positions of the edges.
        goal = (
            np.random.random()*2*PI, #Angle
            150, 150, # poosition of the body
        )

        ## Generate the resources for each agent to play the game.
        resources = []
        for _ in range(len(Agents)):
            resources.append(initResourcers(goal))


        ## Give each of the some time to try and reach the goal state.
        ## Calculate the score for each agent
        # pool =  Pool(processes=Population)
        # results = [pool.apply_async(playOne, args=(Agents[idx], resources[idx],)) for idx in range(Population)]
    
        # ## Assign scores to the agents as they finish.
        # for resIdx in range(len(results)):
        #     score = results[resIdx].get()
        #     Agents[resIdx][1] = score
        # pool.close()
        # pool.join()

        ## Just a single thread
        for idx in range(Population):
            Agents[idx][1] = playOne(Agents[idx], resources[idx])


        ## Sort the agents based on the scores and then generate the children.
        Agents.sort(key=lambda x: x[1], reverse=True)
        parents = Agents[-ParentsCount:]

        if generation%10 == 0:
            for parent in parents:
                parent[0].save("./TEST" + str(generation))

        ## Print the scores of the parents
        print("Top players: ",list(map(lambda x: x[1], parents)))

        for pIdx in range(len(parents)):
            parents[pIdx][1] = 0.0

        newChildren = []
        for _ in range(ChildrenCount):
            ## Choose two random parents
            p1, p2 = sample(parents, 2)
            ## Create new child using the parents
            newChild = crossover(p1[0], p2[0])
            ## Mutate it and add it to the children's list
            newChild.mutate(eta)
            newChildren.append([newChild, 0.0])


        # for pIdx in range(len(parents)):
        #     parents[pIdx][0].mutate(eta)

        Agents = newChildren+parents
        eta = eta*0.9
        ## Break only if battery dies but before that save the weights...


if __name__ == "__main__":
    main()