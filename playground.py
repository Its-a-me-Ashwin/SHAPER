import pymunk
import pygame
import pymunk.pygame_util
from Agent.Agent2 import *
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


afterKFrames = 300

# def on_collision_arbiter_begin(arbiter, space, data):
#     print("hello")
#     return True
        # '''
        # The crux of forming the joint.
        # This function does 2 things - 
        #     1. Check how close the polygon is. and 
        #     2. if the polygon is very close, form a pymunk pinJoint.

        # Note: For this to be executed, there needs to be an event waiting for a collision to happen.
        # The 3 lines mentioned below must be replicated for each arm:

        # collision = space.add_collision_handler(40, 20) #hard coded these since by default the collision type of arm and polygon is set to 20 and 40 respectively.
        # collision.begin = arm.on_collision_arbiter_begin #arm is an object of class Arm1
        # collision.data["polygon"] = polygon.body # polygon is an object of Polygon
        # collision.data["arms_data"] = armData # armData is a disctionary that contains the information about arms.
        #     e.g. armData = {"Arm_1": (arm1.Objects[-1]), "Arm_2": (arm2.Objects[-1])}

        # These lines must be present in the main file that is taking care of the simulation. 
        # Reason: on_collision_arbiter_begin must be called after the event has been triggered.
        # '''
        # polygon = data["polygon"]
        # closest_arm = None
        # min_distance = float('inf')
        # armObject = None

        # for key in data.get("arms_data"):
        #     contact_point = arbiter.contact_point_set.points[0].point_a
        #     temp = data.get("arms_data")[key]
        #     current_arm = data.get("arms_data")[key].Objects[-1]["Object"]
        #     current_arm_length = data.get("arms_data")[key]["Length"]
        #     end_of_current_arm = current_arm.position + (0, current_arm_length / 2)
        #     distance = sqrt(
        #         (end_of_current_arm.x - contact_point.x) ** 2 + (end_of_current_arm.y - contact_point.y) ** 2
        #     )
    
        #     if distance < min_distance:
        #         min_distance = distance
        #         closest_arm = current_arm
        #         closest_arm_length= current_arm_length
        #         armObject = temp
        
        # if armObject.pinJoint!=None and closest_arm is not None:
        #     armObject.pinJoint = pymunk.PinJoint(closest_arm, polygon.body, (0, closest_arm_length / 2),
        #                             polygon.body.world_to_local(contact_point))
        #     space.add(armObject.pinJoint)


## Function in which all the environment is simulated. 
def run(window, space, width=WIDTH, height=HEIGHT):
    run = True
    clock = pygame.time.Clock()
    
    addFloor(space)

    ## Make an agegnt.
    # agent = Agent()

    ## Add all the layers as required.
    # agent.addLayer("Input", 54, None, False)
    # agent.addLayer("H1", 256, Sigmoid, False)
    # agent.addLayer("H2", 128, Sigmoid, False)
    # agent.addLayer("H3", 64, Sigmoid, False)
    # agent.addLayer("Output", 9, None, True)

    ## Every 10 frames the agent gives an output to the engine.
    agentActive = 60


    ## The object that needs to be grabbed and fondled
    polygon = Polygon(space, (10,10), (0,0), [[100, 100], [200, 100], [200, 200]])

    arms = []

    arm1 = Arm1(space, (250, 250))
    arm1.addJoint(100)
    arm1.addJoint(50)
    arm1.addJoint(50,rotation=0.6, end=True)

    arm2 = Arm1(space, (350, 250),2)
    arm2.addJoint(150)
    arm2.addJoint(100)
    arm2.addJoint(50,rotation=0.6, end=True)


    arm3 = Arm1(space, (500, 50),3)
    arm3.addJoint(250)
    arm3.addJoint(150)
    arm3.addJoint(100,rotation=0.6, end=True)

    arms = [arm1, arm2, arm3]

    armData = {"Arm_1": arm1, "Arm_2": arm2, "Arm_3": arm3}
    collision = space.add_collision_handler(40, 20) #hard coded these since by default the collision type of arm and polygon is set to 20 and 40 respectively.
    collision.begin = polygon.on_collision_arbiter_begin #arm is an object of class Arm1
    
    
    ## There is a problem. If we render every frame it looks janky. 
    ## Fix is to run the physics engine at 600Hz and render the stuff at 60Hz.
    ## We also need to set a rate for the AI to run in the background. When the agent responds the physics engine will react.

    frameNumber = 0

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        collision.data["polygon"] = polygon.body # polygon is an object of Polygon
        collision.data["arms_data"] = armData # armData is a disctionary that contains the information about arms.
        # if frameNumber%agentActive == 0:
        #     inputVector = []
        #     for arm in arms:
        #         lTempData = arm.physicsToAgent()
        #         inputVector.extend(lTempData["Angles"])
        #         inputVector.extend(lTempData["Rates"])
        #         inputVector.extend(lTempData["Positions"])
        #     inputVector = np.array(inputVector)
        #     rawOut = agent.forwardPass(inputVector)
        #     print("Out:", rawOut)
        #     ## Use the agents output to manimulate the arms.
        #     k = 0
        #     for arm in arms:
        #         newAngles = []
        #         for idx in range(len(arm.Objects)):
        #             newAngles.append((rawOut[k]+1)*(PI/2))
        #             k+=1 
        #         arm.setAngles(newAngles)

        ## Render only some of the frames. Makes it more smoother.
        for x in range(PHYSICS_FPS):
            space.step(DT/float(PHYSICS_FPS))

        
        # for arm in arms:
        #     arm.getAngles()
        polygon.draw(window)
        draw(space, window, draw_options)
        clock.tick(10)

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