import sys
from typing import List
from Physics.arm import *
from Physics.utils import *
from Agent.agent import *
import random
from tqdm import tqdm

import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d


## Creates the agent with a specific shape and size.
def createAgent():
    lAgent = Agent()
    lAgent.addLayer("Input", 28, None, False)
    lAgent.addLayer("H0", 100, Linear, False)
    lAgent.addLayer("Output", 3, Linear, True)

    return lAgent
    

## Creates the polygons that will be lanuched at the arms.
## The positon and power will be set later.
def createPolygon():
    vs = [(-30, 0), (0, 3), (10, 0), (0, -3)]
    # mass = 1
    # moment = pymunk.moment_for_poly(mass, vs)
    arrow_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    arrow_shape = pymunk.Poly(arrow_body, vs)
    arrow_shape.friction = 0.5
    arrow_shape.collision_type = 1
    arrow_shape.density = 0.1
    return arrow_body, arrow_shape


## Grabbing function. If enabled, will allow the arm to hold on to the polygons. 
def stick_arrow_to_target(space, arrow_body, target_body, position, flying_arrows):
    pivot_joint = pymunk.PivotJoint(arrow_body, target_body, position)
    phase = target_body.angle - arrow_body.angle
    gear_joint = pymunk.GearJoint(arrow_body, target_body, phase, 1)
    space.add(pivot_joint)
    space.add(gear_joint)
    try:
        flying_arrows.remove(arrow_body)
    except:
        pass


## Defing the scoring functions.
## Using this alone cause the bot to become a properllor or vibrator.
def scoreOnlyOnArmContact(collisionShape, arrowShape, data):
    if collisionShape in data["ArmShapes"]:
        return +1.0
    return 0.0

def scoreOnSafeWallHit(collisionShape, arrowShape, data):
    if collisionShape in data["ArmShapes"]:
        #print("Hit the arm")
        return 10
    if collisionShape in data["TargetWall"]:
        #print("Hit the target wall")
        return -250
    if collisionShape in data["SafeWall"]:
        #print("Hit the safe wall")
        return 300
    if collisionShape in data["OtherWalls"]:
        #print("Hit the otherwall")
        pass
    if collisionShape in data["SafeWall1"]:
        #print("Hit the otherwall")
        return 50
    
    if collisionShape == data["Cannon"]:
        return 250
    return 0.0


## Collision Handler.
def post_solve_arrow_hit(arbiter, space, data):
    if arbiter.total_impulse.length > 300:
        a, b = arbiter.shapes
        position = arbiter.contact_point_set.points[0].point_a
        #b.collision_type = 0
        #b.group = 1
        b.color = (255, 255, 0, 255)
        other_body = a.body
        arrow_body = b.body

        data["score"] += scoreOnSafeWallHit(a, b, data)

        if a in data["SafeWall"]:
            space.remove(b)
            data["isThereAFlyingPolygon"] = False
            #print("Hit a safe wall. Score the agent")
        
        if a in data["TargetWall"]:
            #data["score"] -= 1.0
            space.remove(b)
            data["isThereAFlyingPolygon"] = False
            #print("Hit a target wall. Punish the aagent")

        if a in data["OtherWalls"]:
            space.remove(b)
            data["isThereAFlyingPolygon"] = False
            #print("Hit a unnecessary wall. No points")

        if a in data["ArmShapes"]:
            #print("Hit an arm section. Give some score.")
            pass
        if a == data["Cannon"]:
            print("Perfect strike. Give a huge reward.")


## Input to the agent. Extracted from the environment.       
def dataForAgent(polygon):
    return [
        polygon.position[0]/WIDTH,
        polygon.position[1]/HEIGHT,
        polygon.angle/(2*PI),
        polygon.angular_velocity,
        polygon.velocity[0]/2500,
        polygon.velocity[1]/2500
        ## Add velocity too
    ]


offset = 300

## Used for tesing a specific case.
def getKnownPosition():
    return offset, HEIGHT//2-100, WALLRIGHT

def getRandomPositionForCannon():
    line = random.random()

    # if line < 0.25:
    #     return offset + random.random()*(WIDTH-offset),offset, WALLBOTTOM
    if line < 0.5:
        return (WIDTH-offset), random.randint(offset, HEIGHT-offset), WALLLEFT
    # elif line < 0.75:
    #     return offset + random.random()*(WIDTH-offset), (HEIGHT-offset), WALLTOP
    else: 
        return offset, random.randint(offset, HEIGHT-offset), WALLRIGHT

## If display is true the game will be displayed
## If the agent is true we use the given agent to play the game, else the game will without any inputs
## If path is not none we will load the weights from the disk.
def play(display=True, agent=None, path=None, scoreFrameFunc=lambda:0, scoreFullGameFunc= lambda:0, pipeCom=None):
    maxFrames = 60*120

    if agent != None and path != None:
        agent.load(path)

    ### PyGame init
    if display:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 16)
    running = True
    

    ### Init the space
    space = pymunk.Space()
    space.gravity = 0, 500
    if display:
        draw_options = pymunk.pygame_util.DrawOptions(screen)

    ### Init the arms
    arm1 = Arm(space, (WIDTH/2, HEIGHT/2))
    arm1.addJoint(175)
    arm1.addJoint(125)
    arm1.addJoint(50, end=True)

    ### Init the box.
    ### Left, Top, Right, Bottom
    static: List[pymunk.Shape] = [
        pymunk.Segment(space.static_body, (50, HEIGHT-50), (50, 50), 5),
        pymunk.Segment(space.static_body, (50, 50), (WIDTH-50, 50), 5),
        pymunk.Segment(space.static_body, (WIDTH-50, 50), (WIDTH-50, HEIGHT-50), 5),
        pymunk.Segment(space.static_body, (50, HEIGHT-50), (WIDTH-50, HEIGHT - 50), 5),
    ]

    safeWallsExtras: List[pymunk.Shape] = [
        #pymunk.Segment(space.static_body, (1000, 100), (1000, 250), 5),
        pymunk.Segment(space.static_body, (1100, 250), (1300, 250), 5),
        pymunk.Segment(space.static_body, (1300, 250), (1300, 100), 5),
        pymunk.Segment(space.static_body, (1100, 100), (1300, 100), 5),
    ]


    # "Cannon" that can fire arrows
    cannon_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    cannon_shape = pymunk.Circle(cannon_body, 25)
    #cannon_shape.sensor = True
    cannon_shape.color = (255, 50, 50, 255)
    cannon_body.position = 100, 500
    space.add(cannon_body, cannon_shape)

    ## Set the cannon position to a random position on the perimeter of the area.
    originalCannonPos = getKnownPosition()
    cannon_body.position = originalCannonPos[:-1]

    ## Set the target wall. This is based on the posiition of the cannon.
    ## The target wall is the wall that the polygons will try to hit
    targetWall = originalCannonPos[-1]
    static[wallKeyToIdx(targetWall)].color = (255, 0, 0, 255)

    ## Set the safe wall.
    ## This is the wall that the agent will try and deflect the arms to.
    safeWall = oppositeWall(targetWall)
    static[wallKeyToIdx(safeWall)].color = (0, 255, 0, 255)

    ## Set the other two walls.
    otherWalls = static.copy()
    otherWalls.remove(static[wallKeyToIdx(safeWall)])
    otherWalls.remove(static[wallKeyToIdx(targetWall)])

    ## Add the walls to the space
    for s in static:
        s.friction = 1.0
        s.group = 1
    space.add(*static)

    for s in safeWallsExtras:
        s.color = (0, 255, 0, 255)
        s.friction = 1.0
        s.group = 1
    space.add(*safeWallsExtras)

    ## The polygon that will be firied.
    arrow_body, arrow_shape = createPolygon()
    space.add(arrow_body, arrow_shape)

    ## Active lisit of flying polygons.
    flying_arrows: List[pymunk.Body] = []

    ## The handler will be responsible for collisions and scoring the agent.
    handler = space.add_collision_handler(0, 1)

    ## Set the cannon and the bullets.
    handler.data["flying_arrows"] = flying_arrows
    handler.data["cannon"] = cannon_body

    ## Set the wall discription
    ## Can add more safe and target walls as you need
    handler.data["TargetWall"] = [static[wallKeyToIdx(targetWall)]]
    handler.data["SafeWall"] = safeWallsExtras
    handler.data["SafeWall1"] = [static[wallKeyToIdx(safeWall)]]
    handler.data["OtherWalls"] = otherWalls

    ## Add the arm shape to the handler's data.
    handler.data["ArmShapes"] = [x["Shape"] for x in arm1.Objects]
    handler.data["Cannon"] = cannon_shape
    ## Init the score as 0 for all agents
    handler.data["score"] = 0.0

    handler.data["isThereAFlyingPolygon"] = False

    handler.post_solve = post_solve_arrow_hit

    frameNumber = 0
    activePolygon = None
    gammaCorrection = 10*(PI/180)
    dNum = 0
    while running:
        if frameNumber > maxFrames:
            running = False
        if handler.data["isThereAFlyingPolygon"] == False:
            activePolygon = None
        if display:
            for event in pygame.event.get():
                if (
                    event.type == pygame.QUIT
                    or event.type == pygame.KEYDOWN
                    and (event.key in [pygame.K_ESCAPE, pygame.K_q])
                ):
                    running = False
        if not handler.data["isThereAFlyingPolygon"]:
            ## Choose a random angle. But make sure it is aimed at the target wall.

            ## Choose a random angle. But make sure it is aimed at the target wall.

            #cannon_body.angle = DegreesToRads(random.randint(-30, 10))
            #cannon_body.angle = angleList[dNum]
            # angleList.append(cannon_body.angle)
            if targetWall == WALLLEFT:       
                cannon_body.angle = DegreesToRads(random.randint(160, 220))
            if targetWall == WALLRIGHT:       
                cannon_body.angle = DegreesToRads(random.randint(-60, 20))
            # move the unfired arrow together with the cannon
            arrow_body.position = cannon_body.position + Vec2d(
                cannon_shape.radius + 40, 0
            ).rotated(cannon_body.angle)
            arrow_body.angle = cannon_body.angle

            ## Give a random ammout on impulse.
            power = 7500 + random.random()*5000
            #power = powerList[dNum]*5000 + 7500
            impulse = float(power) * Vec2d(1, 0)
            impulse = impulse.rotated(arrow_body.angle)
            arrow_body.body_type = pymunk.Body.DYNAMIC
            arrow_body.apply_impulse_at_world_point(impulse, arrow_body.position)
            #powerList.append((power-7500)/5000)

            # space.add(arrow_body)
            flying_arrows.append(arrow_body)
            handler.data["flying_arrows"] = flying_arrows
            activePolygon = arrow_body
            arrow_body, arrow_shape = createPolygon()
            space.add(arrow_body, arrow_shape)

            ## Indicates a polygon is still in flight.
            handler.data["isThereAFlyingPolygon"] = True
            dNum += 1
            dNum = dNum % 27

        if activePolygon != None:
            inputVector = []
            ## Get the scaled data from the polygon.
            inputVector += dataForAgent(activePolygon)
            ## Get the scaled data from the arm
            armdData = arm1.physicsToAgent()
            #print(armdData)
            for key in armdData:
                inputVector += armdData[key]
            ## TODO Add wall data
            inputVector.append(cannon_body.angle)
            inputVector.extend([cannon_body.position[0],cannon_body.position[1]])
            inputVector.append((power-7500)/5000)
            inputVector = np.array(inputVector)
            inputVector = np.clip(inputVector, a_min=-1, a_max=1)/10
  
            if agent != None:
                rawOut = agent.forwardPass(inputVector)
                output = np.clip(rawOut*1.1, a_min=-10, a_max=10)
                #print(output)
                arm1.agentToPhysicsLocation(output, maxSpeed=1.2)


        for flying_arrow in flying_arrows:
            drag_constant = 0.0002

            pointing_direction = Vec2d(1, 0).rotated(flying_arrow.angle)
            # print(pointing_direction.angle, flying_arrow.angle)
            flight_direction = Vec2d(*flying_arrow.velocity)
            flight_direction, flight_speed = flight_direction.normalized_and_length()

            dot = flight_direction.dot(pointing_direction)
            # (1-abs(dot)) can be replaced with (1-dot) to make arrows turn
            # around even when fired straight up. Might not be as accurate, but
            # maybe look better.
            drag_force_magnitude = (
                (1 - abs(dot)) * flight_speed ** 2 * drag_constant * flying_arrow.mass
            )
            arrow_tail_position = flying_arrow.position + Vec2d(-50, 0).rotated(
                flying_arrow.angle
            )
            flying_arrow.apply_impulse_at_world_point(
                drag_force_magnitude * -flight_direction, arrow_tail_position
            )

            flying_arrow.angular_velocity *= 0.5

        if display:
            screen.fill(pygame.Color("black"))
            space.debug_draw(draw_options)
            pygame.display.flip()

        ### Update physics
        fps = 120
        dt = 1.0 / fps
        space.step(dt)

        if display:
            clock.tick(fps)
        frameNumber += 1
    # angleList = np.array(angleList)
    # powerList = np.array(powerList)

    # np.save("./angle", angleList)
    # np.save("./power", powerList)

    if pipeCom==None:
        return handler.data["score"]
    else:
        pipeCom.send(handler.data["score"])

PopulationCount = 16
ChildrenCount = 10
ParentsCount = PopulationCount-ChildrenCount


## use this function to play with an agent that is trained.
def playGivenAgent(path):
    lAgent= createAgent()

    lAgent.load(path)
    play(display=True, agent=lAgent)

## Example useage:
#sys.exit(playGivenAgent("./TEST_349"))


## Main function written as a single process. If yo computer too slow use this.
def singleMain():
    Agents = []
    avgScores = []
    for _ in range(PopulationCount):
        lAgent= createAgent()
        Agents.append([lAgent, 0.0])

    eta = 0.9
    gamma = 0.01
    generation = 0

    while True:
        for idx in tqdm(range(PopulationCount)):
            Agents[idx][1] = play(display=True, agent=Agents[idx][0])

        Agents.sort(key= lambda x: x[1])

        if (generation+1)%50 ==0:
            Agents[-1][0].save("Test_" + str(generation))

        avgScore = sum(list(map(lambda x: x[1], Agents)))/len(Agents)
        avgScores.append(avgScore)
        print("Scores: ", list(map(lambda x: x[1], Agents)),
              "Average score:", avgScore   
            )

        Parents = Agents[-ParentsCount:]
        Children = []

        for idx in range(ChildrenCount):
            lParents = random.sample(Parents, 2)
            lNewChild = singlePointCrossover(*list(map(lambda x: x[0], lParents)))
            lNewChild.mutate(eta=eta, gamma=gamma)
            Children.append([lNewChild, 0.0])

        eta = eta*0.95
        gamma = gamma*0.9

        eta = max(eta, 0.5)
        gamma = max(0.001, gamma)
        Agents = []

        Agents = []
        Agents = Parents + Children
        generation += 1

        if generation%10 == 0:
            plt.plot(avgScores)
            plt.savefig("SINGLEPROC"+str(generation)+".png")


## Fancy multi process main function. Same as the above function but gives each agent/environment pair a separate process. 
## Ich bin schnell. Schnell wie fick. 
def multiProcessingMain():
    import multiprocessing as mp

    Agents = []
    avgScores = []

    for _ in range(PopulationCount):
        lAgent= createAgent()
        Agents.append([lAgent, 0.0])
    eta = 0.9
    gamma = 0.01
    generation = 0
    while True:
        pipes = []
        processes = []

        for idx in range(PopulationCount):
            p_conn, c_conn = mp.Pipe()
            pipes.append([p_conn, c_conn])
            process = mp.Process(target=play, args=(False, Agents[idx][0], None, None, None, c_conn,))
            process.start()
            processes.append(process)

        for idx in range(PopulationCount):
            p_conn = pipes[idx][0]
            Agents[idx][1] = p_conn.recv()
            processes[idx].join()

        Agents.sort(key= lambda x: x[1])

        if (generation+1)%50 ==0:
            Agents[-1][0].save("Test_" + str(generation))

        avgScore = sum(list(map(lambda x: x[1], Agents)))/len(Agents)
        avgScores.append(avgScore)
        print("Scores: ", list(map(lambda x: x[1], Agents)),
              "Average score:", avgScore   
            )

        Parents = Agents[-ParentsCount:]
        Children = []

        for idx in range(ChildrenCount):
            lParents = random.sample(Parents, 2)
            lNewChild = singlePointCrossover(*list(map(lambda x: x[0], lParents)))
            lNewChild.mutate(eta=eta, gamma=gamma)
            Children.append([lNewChild, 0.0])

        eta = eta*0.95
        gamma = gamma*0.9

        eta = max(eta, 0.5)
        gamma = max(0.001, gamma)
        Agents = []
        Agents = Parents + Children
        generation += 1

        if generation%100 == 0:
            plt.plot(avgScores)
            plt.savefig("MULTIPROC_"+str(generation)+".png")

## Currently, we should monitor the training and change the scoring function when the accuracy starts to plataue.
from matplotlib import pyplot as plt
if __name__ == "__main__":
    lAgent= createAgent()
    play(agent=lAgent)


    # try:
    #     multiProcessingMain()
    # except KeyboardInterrupt:
    #     sys.exit()