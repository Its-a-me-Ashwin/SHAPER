import pymunk
import pygame
import pymunk.pygame_util
from Agent.Agent2 import *
from Physics.arm import *
from Physics.arm2 import *
from Physics.polygon import *
from Physics.utils import *

PHYSICS_FPS = 25


## Setup the envirnment.
def setup():
    pygame.init()

    window = pygame.display.set_mode((WIDTH, HEIGHT))
    space = pymunk.Space()
    space.gravity = (0, 1000)

    draw_options = pymunk.pygame_util.DrawOptions(window)

    ## Window is the pygame window used to render the objects
    ## Space is the pymunk saimulation space. All the physics are calculated in this space.
    ## Draw_options connects the above two so that the space can be rendered in the window.
    return window, space, draw_options


## Function in which all the environment is simulated. 
def run(window, space, width=WIDTH, height=HEIGHT):
    run = True
    clock = pygame.time.Clock()
    
    ## Make an agegnt.
    agent = Agent()

    ## Add all the layers as required.
    agent.addLayer("Input", 59, None, False)
    agent.addLayer("H0", 128, TanH, False)
    agent.addLayer("H1", 256, TanH, False)
    agent.addLayer("H2", 128, TanH, False)
    agent.addLayer("H3", 64, TanH, False)
    agent.addLayer("Output", 9, None, True)

    agent.load("./TEST10")

    ## Every 10 frames the agent gives an output to the engine.
    agentActive = 10


    ## The object that needs to be grabbed and fondled
    polygon = Polygon(space, (10,10), (0,0), [[150, 100], [250, 100], [250, 200]])

    goalState = (
            np.random.random()*2*PI, #Angle
            150, 150, # poosition of the body
    )

    arms = []

    arm1 = Arm1(space, (250, 250))
    arm1.addJoint(100)
    arm1.addJoint(50)
    arm1.addJoint(50, True)

    arm2 = Arm1(space, (750, 250),2)
    arm2.addJoint(150)
    arm2.addJoint(100)
    arm2.addJoint(50, True)


    arm3 = Arm1(space, (500, 50),3)
    arm3.addJoint(250)
    arm3.addJoint(150)
    arm3.addJoint(100, True)

    arms = [arm1, arm2, arm3]


    ## There is a problem. If we render every frame it looks janky. 
    ## Fix is to run the physics engine at 600Hz and render the stuff at 60Hz.
    ## We also need to set a rate for the AI to run in the background. When the agent responds the physics engine will react.

    frameNumber = 0

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        if frameNumber%agentActive == 0:
            ## Agent only controls the arm every 10 frames.
            inputVector = []
            ## Input involving the arms.
            for arm in arms:
                lTempData = arm.physicsToAgent()
                inputVector.extend(lTempData["Angles"])
                inputVector.extend(lTempData["Rates"])
                inputVector.extend(lTempData["Positions"])
            
            ## Input involving the polygon.
            body = polygon.body
            polygonData = [body.position[0], body.position[1]]
            inputVector.extend(polygonData)
            
            ## Input regarding the goal.
            inputVector.extend(goalState)

            inputVector = np.array(inputVector)

            ## Get the prediction from the agent.
            rawOut = agent.forwardPass(inputVector)
            ## Use the agents output to manipulate the arms.
            k = 0
            for arm in arms:
                newRates = []
                for _ in range(len(arm.Objects)):
                    newRates.append(rawOut[k]*4)
                    k+=1
                print("Rates:", newRates)
                arm.agentToPhysics(newRates)

        ## Render only some of the frames. Makes it more smoother.
        for _ in range(PHYSICS_FPS):
            space.step(DT/float(PHYSICS_FPS))
        
        
        polygon.draw(window)
        draw(space, window, draw_options)
        clock.tick(FPS)
        frameNumber += 1

pygame.quit()

## Only call this function when we need to render the objects. In most cases rendering it is not needed.
## Call it every frame that needs to be rendered.
def draw(space, window, draw_options):
    window.fill((255,255,255))
    space.debug_draw(draw_options)
    pygame.display.update()

if __name__ == "__main__":
    window, space, draw_options = setup()
    run(window, space)