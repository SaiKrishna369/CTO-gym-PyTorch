import gym
import random
import numpy as np
from math import sqrt
from gym.envs.classic_control import rendering
from gym import logger

"""
CTO variant with only 1 observer
"""

class CtoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    """
    Summary of the the environment variables
            
        runTime
            The total time simulation runs for

        updateRate
            Time interval between each decision making/action

        episodes
            Equal to runTime / updaRate
            Simulation terminates after these many episodes

        gridWidth, gridHeight
            Dimensions of the 2D arena

        sensorRange
            The maximum distance between agent and target for the agent to notice
            the target
    
    """

    def __init__(self):
        self.viewer = None


    def initialize(self, targets=10, sensorRange=15, updateRate=10, targetMaxStep=100,
                    targetSpeed=1.0,
                    totalSimTime=1500, gridWidth=150, gridHeight=150, compact=False):
        #System variables
        self.curr_episode = 0
        self.curr_step = 0
        
        # general variables in the environment
        self.runTime = totalSimTime        

        #total number of targets in the simulation
        self.numTargets = targets

        #maximum time for which one target can stay oncourse for its destination
        self.targetMaxStep = targetMaxStep

        #speed of target
        self.targetSpeed = targetSpeed
        self.agentSpeed = 1.0

        #sensor range of the observer
        self.sensorRange = sensorRange

        #time after which observer takes the decision
        self.updateRate = updateRate

        #2D field dimensions
        self.gridHeight = gridHeight
        self.gridWidth = gridWidth

        self.compactRepresentation = compact

        #Initialize target locations and their destinations
        self.targetLocations = np.array([[0.0, 0.0]]*self.numTargets)
        self.targetDestinations = np.array([[0.0, 0.0]]*self.numTargets)
        self.targetSteps = np.array([self.targetMaxStep]*self.numTargets)
        self.targetPosIncrements = np.array([(-1000.0, -1000.0)]*self.numTargets)

        for i in xrange(self.numTargets):
            self.targetDestinations[i][0] = random.uniform(0, self.gridWidth)
            self.targetDestinations[i][1] = random.uniform(0, self.gridHeight)

            self.targetLocations[i][0] = random.uniform(0, self.gridWidth)
            self.targetLocations[i][1] = random.uniform(0, self.gridHeight)

            while not self.acceptable(i):
                self.targetLocations[i][0] = random.uniform(0, self.gridWidth)
                self.targetLocations[i][1] = random.uniform(0, self.gridHeight)

        #Initialize the agent and ensure it is not on top of other target
        self.agentPosition = np.array([0.0, 0.0])
        self.agentPosition[0] = random.uniform(0, self.gridWidth)
        self.agentPosition[1] = random.uniform(0, self.gridHeight)
        while not self.acceptable(-1, True):
            self.agentPosition[0] = random.uniform(0, self.gridWidth)
            self.agentPosition[1] = random.uniform(0, self.gridHeight)

        self.agentPosIncrements = np.array([-1000.0, -1000.0])

        self.episodes = self.runTime / self.updateRate  


    # Checks whether the two points are at least one unit apart
    def acceptable(self, index, agent=False):
        if not agent:
            if index == 0:
                return True            
            else:
                for i in xrange(index):
                    if self.distance(self.targetLocations[index], self.targetLocations[i]) <= 1:
                        return False
                return True        
        else:
            for i, pos in enumerate(self.targetLocations):
                if self.distance(self.agentPosition, pos) <= 1:
                        return False
            return True


    # Calculates euclidean distance between two points
    def distance(self, pos1, pos2):
        euclideanDistance = (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2
        return sqrt(euclideanDistance)


    def reset(self):
        if self.compactRepresentation:
            self.state = []
            for i, t in enumerate(self.targetLocations):
                if self.distance(self.agentPosition, t) <= self.sensorRange:
                    self.state.append(t)

        else:
            self.state = [[0.0, 0.0]]*self.numTargets

            for i, t in enumerate(self.targetLocations):
                if self.distance(self.agentPosition, t) <= self.sensorRange:
                    self.state[i] = t

        self.state.append(self.agentPosition)
        return np.array(self.state)


    def step(self, action):
        if self.curr_episode > self.episodes:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'initialize()' and 'reset()' once you receive 'done = True'")
            return

        self.curr_episode += 1

        reward = 0
        agentReachedDest = False
        self.agentPosIncrements = np.array([-1000.0, -1000.0])

        for _ in xrange(self.updateRate):
            self.curr_step += 1

            #Move targets
            for i in xrange(self.numTargets):
                self.moveTarget(i)

            #Move agent
            if not agentReachedDest:
                agentReachedDest = self.moveAgent(action)
            else:
                self.agentPosition = action.astype('float32')

            #Calculate reward at this step
            for i, t in enumerate(self.targetLocations):
                if self.distance(self.agentPosition, t) <= self.sensorRange:
                    reward += 1

            if self.viewer is not None:
                self.render()
        
        return self.reset(), reward, self.curr_episode >= self.episodes, {}
            

    def moveTarget(self, idx):
        # Check if this target has been oncourse for max allowed time or it reached its destination
        if self.targetSteps[idx] == 0 or (abs(self.targetDestinations[idx][0] - self.targetLocations[idx][0]) < 1 and 
            abs(self.targetDestinations[idx][1] - self.targetLocations[idx][1]) < 1):
            self.targetDestinations[idx][0] = random.uniform(0, self.gridWidth)
            self.targetDestinations[idx][1] = random.uniform(0, self.gridHeight)            
            #Create new destination and reset step counter to max allowed time and position increments to default   
            self.targetSteps[idx] = self.targetMaxStep
            self.targetPosIncrements[idx] = np.array((-1000.0, -1000.0))

        if self.targetPosIncrements[idx][0] == -1000.0 or self.targetPosIncrements[idx][1] == -1000.0:
            self.targetPosIncrements[idx] = self.calculateIncrements(self.targetLocations[idx], 
                                                                        self.targetDestinations[idx], self.targetSpeed)       

        self.targetLocations[idx] += self.targetPosIncrements[idx]

        if self.targetLocations[idx][0] < 0:
            self.targetLocations[idx][0] = 0
        if self.targetLocations[idx][0] > self.gridWidth:
            self.targetLocations[idx][0] = self.gridWidth
        if self.targetLocations[idx][1] < 0:
            self.targetLocations[idx][1] = 0
        if self.targetLocations[idx][1] > self.gridHeight:
            self.targetLocations[idx][1] = self.gridHeight

        self.targetSteps[idx] -= 1


    def moveAgent(self, dest):
        if self.agentPosIncrements[0] == -1000.0 or self.agentPosIncrements[1] == -1000.0:
            self.agentPosIncrements = self.calculateIncrements(self.agentPosition, dest, self.agentSpeed)
        
        self.agentPosition += self.agentPosIncrements

        if self.agentPosition[0] < 0:
            self.agentPosition[0] = 0
        if self.agentPosition[0] > self.gridWidth:
            self.agentPosition[0] = self.gridWidth
        if self.agentPosition[1] < 0:
            self.agentPosition[1] = 0
        if self.agentPosition[1] > self.gridHeight:
            self.agentPosition[1] = self.gridHeight

        if abs(dest[0] - self.agentPosition[0]) < 1 and abs(dest[1] - self.agentPosition[1]) < 1:
            return True
        else:
            return False

    
    def calculateIncrements(self, loc, dest, speed):
        dx = 1.0*dest[0] - loc[0]
        dy = 1.0*dest[1] - loc[1]

        theta = 0.0
        if abs(dx) > abs(dy):
            theta = abs(dx)
        else:
            theta = abs(dy)
        
        if theta == 0.0:
            return np.array((0.0, 0.0))

        xInc = dx / theta
        yInc = dy / theta
        normalizer = sqrt(xInc**2 + yInc**2)

        xInc = (xInc / normalizer)*speed
        yInc = (yInc / normalizer)*speed

        return np.array((xInc, yInc))

    def getAgentPosition(self):
        return self.agentPosition


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        borderOffset = 50.0 #Reduces 50px along 4 sides

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.scale = ( (screen_width - 2*borderOffset)*1.0/self.gridWidth, 
                    (screen_height - 2*borderOffset)*1.0/self.gridHeight)
            #Borders for neat view
            border1 = rendering.Line((borderOffset, borderOffset), 
                                    (screen_width - borderOffset, borderOffset))
            self.viewer.add_geom(border1)
            border2 = rendering.Line((borderOffset, borderOffset), 
                                    (borderOffset, screen_height - borderOffset))
            self.viewer.add_geom(border2)
            border3 = rendering.Line((screen_width - borderOffset, screen_height - borderOffset), 
                                    (screen_width - borderOffset, borderOffset))
            self.viewer.add_geom(border3)
            border4 = rendering.Line((screen_width - borderOffset, screen_height - borderOffset), 
                                    (borderOffset, screen_height - borderOffset))
            self.viewer.add_geom(border4)

            #Memorize
            self.targets_geom = []
            for i in self.targetLocations:
                point = (self.scale[0]*i[0] + borderOffset, self.scale[1]*i[1] + borderOffset)

                location = rendering.Transform(translation=point)
                axle = rendering.make_circle(4.0)
                axle.add_attr(location)
                axle.set_color(1.0, 0.0, 0.0)
                self.targets_geom.append(location)
                self.viewer.add_geom(axle)

            self.agent_geom = (self.scale[0]*self.agentPosition[0] + borderOffset, 
                                self.scale[1]*self.agentPosition[1] + borderOffset)
            location = rendering.Transform(translation=self.agent_geom)
            self.agent_geom = rendering.make_circle(4.0)
            self.agent_geom.add_attr(location)
            self.agent_geom.set_color(0.0, 0.0, 1.0)
            self.viewer.add_geom(self.agent_geom)
            self.agent_geom = location

            coverage = self.viewer.draw_circle(radius=self.scale[0]*self.sensorRange, res=30, filled=False)
            coverage.add_attr(location)
            coverage.set_color(0.5, 0.5, 0.8)

            self.viewer.add_geom(coverage)

        else:
            for i, t in enumerate(self.targets_geom):
                point = (self.scale[0]*self.targetLocations[i][0] + borderOffset, self.scale[1]*self.targetLocations[i][1] + borderOffset)

                t.set_translation(point[0], point[1])

            point = (self.scale[0]*self.agentPosition[0] + borderOffset, self.scale[1]*self.agentPosition[1] + borderOffset)
            self.agent_geom.set_translation(point[0], point[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def stopRender(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None