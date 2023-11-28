import numpy as np
from math import exp
import pymunk
import random

def TanH(x):
    return np.tanh(x)

## Activation functions
def Sigmoid(x):
    return 2*(1/(1+np.exp(-1*x))-0.5)

## Output is not always between -1 and 1
def ELU(x):
    y = np.where(x<=0, x, 0.1*x)
    return np.where(y>1,y,1)

def Linear(x):
    return x

def ReLu(x):
    return (x+abs(x))/2

def vectorize(mat):
    size = mat.shape[0] * mat.shape[1]
    return np.reshape(mat, size)

def vecToMat(vec, shape):
    return np.reshape(vec, shape)

class Agent:
    def __init__(self):
        ## We use the comple variable to make sure the network is well definied.
        ## Once the network is well definied we allocate the memory and initialize the weights.
        ## complete should be set to True for the network to be used. 
        self.complete = False
        ## This stores the list of matrices that represent the neural network.
        ## This will be initialized when self.complete is moved to True
        self.network = []
        ## Stores the list of activation functions. One per layer. 
        self.activation = []
        ## Store the structure of the network as follows: [{Name: A, Size: x, activation: y}...]
        self.layers = []


    ## Name is a string. 
    ## Size is an integer.
    ## Activation must be a function that takes a single float input and returns a float number.
    ## If output is set to true, we allocate the memory for the network. q  
    def addLayer(self, layerName, size, activation, output=False):
        if self.complete:
            print("Cannot resize a network once memory has been allocated. Delete and create a new agent.")
            return
        if len(self.layers) == 0 or activation == None:
            self.layers.append({"Name": layerName, "Size":size, "Activation": Linear, "Output": output})
        else:
            self.layers.append({"Name": layerName, "Size":size, "Activation": activation, "Output": output})
        if output:
            self._createnetwork()
        
    def _createnetwork(self):
        if len(self.layers) == 2:
            self.network = (np.random.rand(self.layers[0]["Size"],self.layers[1]["Size"])-0.5)*2
            self.activation.append(self.layers[1]["Activation"])
        else:
            for idx in range(len(self.layers)-1):
                self.network.append((np.array(np.random.rand(self.layers[idx]["Size"],self.layers[idx+1]["Size"]))-0.5)*2)
                self.activation.append(self.layers[idx]["Activation"])
        self.complete = True

    def forwardPass (self, inputVector, v=False):
        if self.complete:
            if v:
                print("Input:", inputVector.shape)
                print("Layer 1:", self.network[0].shape, self.activation[0].__name__)
            out = np.matmul(inputVector, self.network[0])
            out = self.activation[0](out)
            for idx in range(1, len(self.network)):
                if v: print("Layer" + str(idx) + ":", self.network[idx].shape, self.activation[idx].__name__)
                out = np.matmul(out, self.network[idx])
                out = self.activation[idx](out)
            if v:
                print(out.shape)
            return out
        else:
            print("Cnnot predict with an incomplete network.")
        return None
    
    ## Might be needed in the future.
    def matToVec(self):
        pass

    def vecToMat(self):
        pass

    def save(self):
        if self.complete:
            ## TODO implement me!!
            pass

    def load(self, path):
        if not self.complete:
            ## TODO Implement me!!
            pass

    def normalize(self, x):
        for idx in range(len(self.network)):
            self.network[idx] = (self.network[idx] - np.min(self.network[idx]))/(np.max(self.network[idx]) - np.min(self.network[idx]))
    
    ## Adds randomness to the network and normalizes the values.
    def mutate(self, eta=0.01):
        for idx in range(len(self.network)):
            np.random.seed(random.randint(1, 1000000))
            rand = (np.array(np.random.random(self.network[idx].shape))-0.5)*2*eta
            self.network[idx] = self.network[idx] + rand
            self.network[idx] = (self.network[idx] - np.min(self.network[idx]))/(np.max(self.network[idx]) - np.min(self.network[idx]))

    ## Just to print the network in a nice way.
    def __repr__(self) -> str:
        out = ""
        for idx in range(len(self.layers)):
            out += "Layer: " + self.layers[idx]["Name"] + " Size: " +  str(self.layers[idx]["Size"]) + "\n"
        out += "Complete: " + str(self.complete) + "\n"
        out += "Length of Networks: " + str(len(self.network)) + "\n"
        out += "Length of Activations: " + str(len(self.activation)) + "\n"
        return out


## Take two agents and returns a new agent that is the combination of both.
## Each matrix is considered as an alle. Each parent contributes half of its alle.
def crossover(agent1, agent2):
    ## Assuming that agent1 and agent2 are of same dimensions.

    ## Create a new agent.
    newAgent = Agent()
    for layerIdx in range(len(agent1.layers)):
        layerDetails = agent1.layers[layerIdx]
        newAgent.addLayer(layerDetails["Name"], layerDetails["Size"], layerDetails["Activation"], layerDetails["Output"])

    ## TODO FIX THIS PLEASE.
    ## Looks like while creating a new child we are missing something.
    numOfMatrices = len(agent1.network)
    for matIdx in range(numOfMatrices):
        shape = agent1.network[matIdx].shape
        agent1Vec = vectorize(agent1.network[matIdx])
        agent2Vec = vectorize(agent2.network[matIdx])

        newVector = []
        for idx in range(agent1Vec.shape[0]):
            if np.random.random() > 0.5:
               newVector.append(agent1Vec[idx])
            else:
                newVector.append(agent2Vec[idx])
        newVector=np.array(newVector)
        newMat = vecToMat(newVector, shape)
        newAgent.network[matIdx] = newMat

    return newAgent

## Take two agents and returns a new agent that is the combination of both.
## Just averages the values of the weights.
def crossoverAvg(agent1, agent2):
    ## Assuming that agent1 and agent2 are of same dimensions.

    ## Create a new agent.
    newAgent = Agent()
    for layerIdx in range(len(agent1.layers)):
        layerDetails = agent1.layers[layerIdx]
        newAgent.addLayer(layerDetails["Name"], layerDetails["Size"], layerDetails["Activation"], layerDetails["Output"])

    numOfMatrices = len(agent1.network)
    for matIdx in range(numOfMatrices):
        newAgent.network[matIdx] = (agent1.network[matIdx]+agent2.network[matIdx])/2

    return newAgent


## We dont render stuff when we train as rendering will slow things down.
def initResourcers(goalState):
    ## Initialize the space.
    space = pymunk.Space()
    space.gravity = (0, 0)

    ## Initialize the arms.
    arms = []
    # Arm 1
    arm1 = Arm1(space, (250, 250))
    arm1.addJoint(100)
    arm1.addJoint(50)
    arm1.addJoint(50, True)
    ## Arm 2
    arm2 = Arm1(space, (750, 250),2)
    arm2.addJoint(150)
    arm2.addJoint(100)
    arm2.addJoint(50, True)
    # Arm 3
    arm3 = Arm1(space, (500, 50),3)
    arm3.addJoint(250)
    arm3.addJoint(150)
    arm3.addJoint(100, True)

    # List of arms.
    arms = [arm1, arm2, arm3]

    ## Initialize the polygon.
    ## Generate a random polygon. For now lets get the triangles working.
    polygon = Polygon(space, (10,10), (0,0), [[150, 100], [250, 100], [250, 200]])

    ## Set the goal state.
    ## Goal state is common for all the agents.
    resources = {
        "Space": space,
        "Arms": arms,
        "Object": polygon,
        "Goal": goalState
    }

    return resources

def scoreFrame(polygon, goalState, scoreData=(100, 0.75)):
    ScoreForAngle = scoreData[0]
    ScoreForPosition = scoreData[1]
    body = polygon.body

    score = abs(body.angle - goalState[0])*ScoreForAngle
    score += abs(body.position[0] - goalState[1])*ScoreForPosition
    score += abs(body.position[1] - goalState[2])*ScoreForPosition

    ## Punish the agent for getting the polygon outside the bounds.
    if abs(body.position[0]) > 2000 or abs(body.position[1]) > 2000:
        score += 1000000

    return score


def playOne(agent, resource, LowPassFilter=0.8, framesPerAgent=120*60, PHYSICS_FPS=25, DT=1/60.0):
    ## Get the resources for the agent
    arms = resource["Arms"]
    space = resource["Space"]
    polygon = resource["Object"]
    goalState = resource["Goal"]

    score = 0
    framNumber = 0
    while framNumber < framesPerAgent:
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
        rawOut = agent[0].forwardPass(inputVector)
        ## Use the agents output to manipulate the arms.
        k = 0
        for arm in arms:
            newAngles = []
            for _ in range(len(arm.Objects)):
                newAngles.append((rawOut[k]+1)*(PI/2))
                k+=1 
            arm.agentToPhysics(newAngles)

        ## Render only some of the frames. Makes it more smoother.
        for _ in range(PHYSICS_FPS):
            space.step(DT/float(PHYSICS_FPS))
        
        ## GetAngles updates the angles of the arm based on the current arm location.
        # for arm in arms:
        #     arm.getAngles()
        
        ## Get score for this perticular frame.
        frameScore = scoreFrame(polygon, goalState)
        ## Low pass filter.
        score = LowPassFilter*score + (1-LowPassFilter)*frameScore
        
        # polygon.draw(window)
        # draw(space, window, draw_options)
        #clock.tick(FPS)
        framNumber += 1
    print("Loss", score)
    return score


import pymunk
from pymunk import SimpleMotor
#from Physics.utils import *
from math import atan2, sin, cos
#from Physics.gripper import *
#from Physics.armsection import *
#from Physics.ball import *


MASS_PER_LENGTH = 10
ARM_WIDTH = 10
PI = 355/113 ## Fancy approximation for Pi


## PID settings
P_ = 1.9
D_ = -0.5
I_ = 0.0005

class Arm1():
    def __init__(self, space, anchorPoistion, group=1):
        self.space = space
        if len(anchorPoistion) != 2:
            print("Error size incorrect")
            return
        self.anchor = anchorPoistion
        self.complete = False

        self.CurrentAngles = []
        self.ExpectedAngles = []

        self.Objects = []
        self.collType = group
        self.shapeFilter = pymunk.ShapeFilter(group=1)


        ## Number of frames since the expected angles were set.
        ## This variable will be rest every time the SetAngles function is called.
        ## It will be incremented every time the arbiter is executed successfully. 
        self.diffCounter = []

    def addJoint(self, length, end=False):
        if self.complete:
            return
        if len(self.Objects) == 0: 
            ## Need to add an anchor.

            ## Generate the anchor.
            newAnchor = pymunk.Body(body_type=pymunk.Body.STATIC)
            newAnchor.position = self.anchor

            ## Generate the mass of the body.
            armMass = MASS_PER_LENGTH*length
            ## Create the body.
            newArmObject = pymunk.Body(armMass, pymunk.moment_for_box(armMass, (ARM_WIDTH, length)))
            newArmObject.position = self.anchor[0], self.anchor[1]+length/2

            ## Create the shape for the object.
            newArmShape = pymunk.Poly.create_box(newArmObject, (ARM_WIDTH, length))
            newArmShape.color = 0, 0, 0, 100

            ## Add constrains and motors to the bodies.
            newJoint = pymunk.PinJoint(newArmObject, newAnchor, (0, -length/2), (0, 0))
            newMotor = pymunk.SimpleMotor(newArmObject, newAnchor, 0.0)

            ## Disable colisions between the arms. 
            ## Might remove this based on hhow the model performs.
            newArmShape.filter = self.shapeFilter

            ## Also add some meta data here. Makes it easier for rendering.
            self.Objects.append({
                                "Object":newArmObject,
                                "Motor": newMotor,
                                "Middle" : (self.anchor[0], self.anchor[1]+length/2),
                                "Length": length
                                })

            ## Add all the objects and shapes to the space.
            self.space.add(newArmObject)
            self.space.add(newArmShape)
            self.space.add(newJoint)
            self.space.add(newMotor)

        else:
            ## Need to add a new arm.

            prevBody = self.Objects[-1].get("Object")
            prevPosition = prevBody.position
            prevLength = self.Objects[-1].get("Length")

            ## Generate the mass of the body.
            armMass = MASS_PER_LENGTH*length
            ## Create the body.
            newArmObject = pymunk.Body(armMass, pymunk.moment_for_box(armMass, (ARM_WIDTH, length)))
            newArmObject.position = prevPosition[0], prevPosition[1]+prevLength/2+length/2

            ## Create the shape for the object.
            newArmShape = pymunk.Poly.create_box(newArmObject, (ARM_WIDTH, length))
            newArmShape.color = 0, 0, 0, 100

            ## Add constrains and motors to the bodies.
            newJoint = pymunk.PinJoint(newArmObject, prevBody, (0, -length/2), (0, prevLength/2))
            newMotor = pymunk.SimpleMotor(newArmObject, prevBody, 0.0)

            ## Disable colision between the arms.
            newArmShape.filter = self.shapeFilter

            ## Also add some meta data here. Makes it easier for rendering.
            self.Objects.append({
                                "Object":newArmObject,
                                "Motor": newMotor,
                                "Middle" : (prevPosition[0], prevPosition[1]+prevLength/2+length/2),
                                "Length": length,
                                "Shape": newArmShape
                            })
            self.space.add(newArmObject)
            self.space.add(newArmShape)
            self.space.add(newJoint)
            self.space.add(newMotor)
        if end:
            self.complete = True
            self.CurrentAngles = [0]*len(self.Objects)
            self.diffCounter = [0]*len(self.Objects)
            #self.space.add_collision_handler()


    ## Might need this function for compatibility with other objects.
    def draw(self, display, color = [0, 0, 0]):
        return
    
    def preHit(self, arbiter, space, data):
        print("Pre hit called for arm")
        pass

    def grab(self):
        ## Find the collision between the last arm segment and the polygon. 
        ## If collision present or distance is small then add a pin constrain on them.
        terminalArmSegment = self.Objects[-1]


    ## Inputs are between -1 and 1, convert it to MAX SPEED
    ## Converts the agent's output to a format that the physics engine can work with.
    def agentToPhysics(self, agentData):
        if len(agentData) != len(self.Objects):
            print("Angent output shape incorrect")
            return
        inputs = list(map(lambda x: ((x+1)/2.0)*4, agentData))
        for idx in range(len(inputs)):
            self.Objects[idx]["Motor"].rate = inputs[idx]
        return inputs

    ## Converts the data from the physcis engine to a format that can be processed by the agent
    ## Normalize the angles and the position of the bodies. Convert to radians if necessary.
    def physicsToAgent(self):
        agentInputs = dict()
        agentInputs["Angles"] = []
        agentInputs["Rates"] = []
        agentInputs["Positions"] = []
        for objIdx in range(len(self.Objects)):
            obj = self.Objects[objIdx]["Object"]
            mot = self.Objects[objIdx]["Motor"]
            l = self.Objects[objIdx]["Length"]/2
            ## The current angle of the arms wrt to the global XY plane
            agentInputs["Angles"].append(obj.angle)
            ## The rate at which the arms are moving
            agentInputs["Rates"].append(mot.rate)
            ## Gets the position of the endpoints of the arm.
            agentInputs["Positions"].extend(centerToEndPoints(obj.position,l,obj.angle))
        return agentInputs

    ## Set the expected angles.
    def setAngles(self, inputs):
        if not self.complete:
            return
        if len(inputs) != len(self.Objects):
            return
        ## Clip it between 0 and 2PI
        inputs = list(map(lambda x: x%(2*PI), inputs))
        self.ExpectedAngles = inputs
        self.diffCounter = [0]*len(self.Objects)

    ## Get the current angles
    def getAngles(self, update=True):
        for objectIdx in range(len(self.Objects)):
            self.CurrentAngles[objectIdx] = self.Objects[objectIdx]["Object"].angle
        if update:
            self.arbiterAgent()
        #print("Current angle:", self.CurrentAngles, "Expected angle:", self.ExpectedAngles)
        return self.CurrentAngles
    
    # Once we get the data from the agent we need the physics engine to execute it.
    # This function uses a simple PID system to achieve that. 
    # DIFF = (EXPECTED - CURRENT)
    def arbiterAgent(self):
        if len(self.ExpectedAngles) == 0:
            return
        diff = list(map(lambda x: x[1]-x[0], zip(self.CurrentAngles, self.ExpectedAngles)))
        for objIdx in range(len(self.Objects)):
            if diff[objIdx] < 10**-6:
                continue
            self.diffCounter[objIdx] += diff[objIdx]
            self.Objects[objIdx]["Motor"].rate = P_*diff[objIdx] + D_*self.Objects[objIdx]["Motor"].rate + I_*self.diffCounter[objIdx]

def centerToEndPoints(centerPos, length, angle):
    return [centerPos[0]+length*cos(angle), centerPos[1]+length*sin(angle),
            centerPos[0]-length*cos(angle), centerPos[1]-length*sin(angle)
            ]


# if __name__ == "__main__":
#     space = pymunk.Space()
#     arm = Arm1(space, (250, 250))
#     arm.addJoint(100)
#     arm.addJoint(100, True)

#     curAngles = arm.getAngles()
#     print("Current Angles:", curAngles)

#     arm.setAngles([3.14, 0])

#     while True:
#         curAngles = arm.getAngles()
#         print("Current Angles:", curAngles)


import pymunk
import pygame
from math import atan2

class Polygon():
    def __init__(self, space, originalAngle, goalAngle, points):
        self.orginal = originalAngle
        self.goal = goalAngle
        self.currentAngle = originalAngle

        self.points = points
        
        positionX = 0
        positionY = 0
        for idx in range(len(points)):
            positionX += points[idx][0]
            positionY += points[idx][1]

        positionX = positionX/len(points)
        positionY = positionY/len(points)

        self.body = pymunk.Body()
        self.body.position = positionX, positionY
        self.body.angle = atan2(originalAngle[1],originalAngle[0])
        self.shape = pymunk.Poly(self.body, self.points)
        self.shape.density = 1
        self.shape.friction = 1
        space.add(self.body, self.shape)


    def draw(self, display, color=(255,255,255)):
        pygame.draw.polygon(display, color, list(map(convertCoordinartes, self.points)))


    ## We need more fucntions like so to extract the required inputs to the agents.
    ## Think of all the details the agent might need reaggarding the object and implement them.
    def getCurrentPosition(self):
        return self.body.position
    
    def getCurrentVelocity(self):
        return self.body.velocity_at_world_point


## Just some testing code.
if __name__ == "__main__":
    ## Make an agegnt.
    a = Agent()

    ## Add all the layers as required.
    a.addLayer("Input", 28, None, False)
    a.addLayer("H1", 150, Sigmoid, False)
    a.addLayer("H2", 50, Sigmoid, False)
    a.addLayer("H3", 20, ReLu, False)
    a.addLayer("Output", 10, None, True)

    ## Generate a random input.
    input = np.random.rand(1,28)
    print(a)

    ## Get an output for the random input.
    out = a.forwardPass(input, True)
    print(out)


    p1 = Agent()

    p1.addLayer("Input", 1, None, False)
    p1.addLayer("H1", 10, Sigmoid, False)
    p1.addLayer("Output", 1, Sigmoid, True)

    p2 = Agent()

    p2.addLayer("Input", 1, None, False)
    p2.addLayer("H1", 10, Sigmoid, False)
    p2.addLayer("Output", 1, Sigmoid, True)

    for _ in range(10):
        newKid = crossover(p1, p2)
        print(newKid.network[0][0])