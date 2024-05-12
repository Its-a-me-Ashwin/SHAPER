import pymunk
from pymunk import SimpleMotor
from math import atan2, sin, cos, sqrt
import numpy as np


MASS_PER_LENGTH = 10
ARM_WIDTH = 10
PI = 355/113 ## Fancy approximation for Pi
HEIGHT = 850
WIDTH = 1500


## PID settings
P_ = 1.9
D_ = -0.5
I_ = 0.0005


## The arm has all the information realted to the arm.
## Allows the agent to control the joints.
class Arm():
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

        # One purpose of the arm is to track the polygon so that it can catch it.
        # This parameters govern it
        self.pinJoint = None

    ## Add new joints as and when we need it. 
    def addJoint(self, length, rotation = 0, collision_type = 20, end=False):
    
        if self.complete:
            # if this is the end of the arm, set a collision type to it so that the gripper can use it.
            # self.Objects[-1]["Shape"].collision_type=collision_type
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
            newArmShape.color = 255, 255, 255, 0

            ## Add constrains and motors to the bodies.
            newJoint = pymunk.PinJoint(newArmObject, newAnchor, (0, -length/2), (0, 0))
            newMotor = pymunk.SimpleMotor(newArmObject, newAnchor, rotation)

            ## Disable colisions between the arms. 
            ## Might remove this based on hhow the model performs.
            newArmShape.filter = self.shapeFilter

            ## Also add some meta data here. Makes it easier for rendering.
            self.Objects.append({
                                "Object":newArmObject,
                                "Motor": newMotor,
                                "Middle" : (self.anchor[0], self.anchor[1]+length/2),
                                "Length": length,
                                "Shape": newArmShape
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
            newArmShape.color = 255, 255, 255, 0

            ## Add constrains and motors to the bodies.
            newJoint = pymunk.PinJoint(newArmObject, prevBody, (0, -length/2), (0, prevLength/2))
            newMotor = pymunk.SimpleMotor(newArmObject, prevBody, rotation)

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
            #self.Objects[-1]["Shape"].collision_type=collision_type
         

            self.complete = True
            self.CurrentAngles = [0]*len(self.Objects)
            self.diffCounter = [0]*len(self.Objects)
            #self.space.add_collision_handler()


    ## Might need this function for compatibility with other objects.
    ## Just to make sure it conforms to the other interfaces.
    def draw(self, display, color = [0, 0, 0]):
        return
    
    def preHit(self, arbiter, space, data):
        print("Pre hit called for arm")
        pass


    ## Converts that sets the motor speeds based on the nt's output.
    ## agentData is a list where values are between -1 and 1
    ## The grab and release mechanisim will be handled in the main routine.
    def agentToPhysics(self, agentData, maxSpeed=4):
        if len(agentData) != len(self.Objects):
            return
        for idx in range(len(agentData)):
            self.Objects[idx]["Motor"].rate = agentData[idx]*maxSpeed
        return
    

    ## Allows the agent to set an angle that the arm will move to.
    ## Has a PID loop to set it self to the correct angle.
    def agentToPhysicsLocation(self, agentData, maxSpeed=4):
        if len(agentData) != len(self.Objects):
            return
        agentData = np.clip(agentData, a_min=-PI, a_max=PI)

        diff = [0]*len(self.Objects)
        mSpeed = [0]*len(self.Objects)

        for idx in range(len(self.Objects)):
            diff[idx] = self.Objects[idx]["Object"].angle - agentData[idx]
            mSpeed[idx] = diff[idx] * P_ + diff[idx] * D_ + diff[idx] * I_


        mSpeed = np.array(mSpeed)
        mSpeed = np.clip(mSpeed, a_min=-maxSpeed, a_max=maxSpeed)

        for idx in range(len(mSpeed)):
            self.Objects[idx]["Motor"].rate = mSpeed[idx]
        return
        
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
            agentInputs["Angles"].append(obj.angle/(2*PI))
            ## The rate at which the arms are moving
            agentInputs["Rates"].append(obj.angular_velocity/4)
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
    
    ## Called by the collision handler.
    def grab(self,anchor,polygon):
        if self.pinJoint is None and self.Objects[-1]["Object"] is not None:
            self.pinJoint = pymunk.PinJoint(self.Objects[-1]["Object"], polygon, (0, self.Objects[-1]["Length"] / 2),
                                    polygon.world_to_local(anchor))
            self.space.add(self.pinJoint)

    def dropPolygon(self):
        '''
        When called, this method removes the pinJoin of the arm if it has any.
        '''

        if self.pinJoint is not None:
            self.space.remove(self.pinJoint)
            self.pinJoint = None

def centerToEndPoints(centerPos, length, angle):
    return [
            (centerPos[0]+length*cos(angle))/WIDTH, (centerPos[1]+length*sin(angle))/HEIGHT,
            (centerPos[0]-length*cos(angle))/WIDTH, (centerPos[1]-length*sin(angle))/HEIGHT
            ]


if __name__ == "__main__":
    space = pymunk.Space()
    arm = Arm(space, (250, 250))
    arm.addJoint(100)
    arm.addJoint(100, True)

    curAngles = arm.getAngles()
    print("Current Angles:", curAngles)

    arm.setAngles([3.14, 0])

    while True:
        curAngles = arm.getAngles()
        print("Current Angles:", curAngles)