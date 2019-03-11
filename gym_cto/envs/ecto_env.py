import gym
import random
import numpy as np
from math import sqrt
from gym.envs.classic_control import rendering
from gym import logger

"""
CTO variant with only multiple observers
"""

class eCtoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None


    def initialize(self, targets=10, agents=10, sensorRange=15, updateRate=10, targetMaxStep=100,
                    targetSpeed=1.0, agentSpeed=1.0,
                    totalSimTime=1500, gridWidth=150, gridHeight=150, compact=False, mark=False):
        #System variables
        self.curr_episode = 0
        self.curr_step = 0
        
        # general variables in the environment
        self.runTime = totalSimTime        

        #total number of targets in the simulation
        self.numTargets = targets

        #total number of agents (observers) in the simulation
        self.numAgents = agents

        #maximum time for which one target can stay oncourse for its destination
        self.targetMaxStep = targetMaxStep

        #speed of target and observer
        self.targetSpeed = targetSpeed
        self.agentSpeed = agentSpeed

        #sensor range of the observer
        self.sensorRange = sensorRange

        #time after which observer takes the decision
        self.updateRate = updateRate

        self.compactRepresentation = compact
        self.markRewardGivingTargets = mark

        #2D field dimensions
        self.gridHeight = gridHeight
        self.gridWidth = gridWidth

        #Initialize target locations and their destinations
        self.targetLocations = np.array([(0.0, 0.0)]*self.numTargets)
        self.targetDestinations = np.array([(0.0, 0.0)]*self.numTargets)
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

        #Initialize the agents and ensure it is not on top of other target or other agents
        self.agentLocations = np.array([(0.0, 0.0)]*self.numAgents)
        self.agentPosIncrements = np.array([(-1000.0, -1000.0)]*self.numAgents)

        for i in xrange(self.numAgents):
            self.agentLocations[i][0] = random.uniform(0, self.gridWidth)
            self.agentLocations[i][1] = random.uniform(0, self.gridHeight)

            while not self.acceptable(i, True):
                self.agentLocations[i][0] = random.uniform(0, self.gridWidth)
                self.agentLocations[i][1] = random.uniform(0, self.gridHeight)

        self.episodes = self.runTime / self.updateRate  


    # Checks whether the two points are at least one unit apart
    def acceptable(self, index, agent=False):
        if not agent: #if only target, just check with other targets
            if index == 0:
                return True            
            else:
                for i in xrange(index):
                    if self.distance(self.targetLocations[index], self.targetLocations[i]) <= 1:
                        return False
                return True        
        else: #first check with targets and then the other agents
            for i, pos in enumerate(self.targetLocations):
                if self.distance(self.agentLocations[index], pos) <= 1:
                    return False
            
            for i in xrange(index):
                if self.distance(self.agentLocations[index], self.agentLocations[i]) <= 1:
                    return False
            return True


    # Calculates euclidean distance between two points
    def distance(self, pos1, pos2):
        euclideanDistance = (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2
        return sqrt(euclideanDistance)


    def reset(self):
        _, reward_assigned_to = self.calculateAgentRewards()
        self.state = []

        if self.compactRepresentation:
            for i in xrange(self.numAgents):
                agent_state = []
                for j, t in enumerate(self.targetLocations):
                    if self.distance(self.agentLocations[i], t) <= self.sensorRange:
                        if self.markRewardGivingTargets:
                            if reward_assigned_to[j] == i:
                                agent_state.append([t[0], t[1], 1, 1])
                            else:
                                agent_state.append([t[0], t[1], 1, 0])
                        else:
                            agent_state.append([t[0], t[1], 1])

                for j in xrange(self.numAgents):
                    if self.distance(self.agentLocations[i], self.agentLocations[j]) <= self.sensorRange:
                        if self.markRewardGivingTargets:
                            agent_state.append([self.agentLocations[j][0], self.agentLocations[j][1], 2, 0])
                        else:
                            agent_state.append([self.agentLocations[j][0], self.agentLocations[j][1], 2])
                
                self.state.append(agent_state)            
        else:            
            for i in xrange(self.numAgents):
                agent_state = []
                if self.markRewardGivingTargets:
                    agent_state = [[0.0,0.0,0]]*(self.numTargets + self.numAgents)
                else:
                    agent_state = [[0.0,0.0]]*(self.numTargets + self.numAgents)

                for j, t in enumerate(self.targetLocations):
                    if self.distance(self.agentLocations[i], t) <= self.sensorRange:
                        if self.markRewardGivingTargets:
                            if reward_assigned_to[j] == i:
                                agent_state[j] = [t[0], t[1], 1]
                            else:
                                agent_state[j] = [t[0], t[1], 0]
                        else:
                            agent_state[j] = t

                for j in xrange(self.numAgents):
                    if self.distance(self.agentLocations[i], self.agentLocations[j]) <= self.sensorRange and j != i:
                        if self.markRewardGivingTargets:
                            agent_state[self.numTargets + j] = (self.agentLocations[j][0], self.agentLocations[j][1], 0)
                        else:
                            agent_state[self.numTargets + j] = self.agentLocations[j]
                
                self.state.append(agent_state)

        return np.array(self.state)


    def step(self, action):
        if self.curr_episode > self.episodes:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'initialize()' and 'reset()' once you receive 'done = True'")
            return

        action = np.array(action)
        if action.shape != (self.numAgents, 2):
            logger.error("Incorrect dimenions of action. Action must have destination position for each agent")
            return

        self.curr_episode += 1

        reward = np.zeros(self.numAgents)
        self.agentPosIncrements = np.array([(-1000.0, -1000.0)]*self.numAgents)
        agentReachedDest = [False]*self.numAgents
        for _ in xrange(self.updateRate):
            self.curr_step += 1

            #Move targets
            for i in xrange(self.numTargets):
                self.moveTarget(i)

            #Move agent
            for i in xrange(self.numAgents):
                if not agentReachedDest[i]:
                    agentReachedDest[i] = self.moveAgent(i, action[i])
                else: #Already reached. Removes precision errors
                    self.agentLocations[i] = action[i].astype('float32')

            #Calculate reward at this step
            reward += self.calculateAgentRewards()[0]

            if self.viewer is not None:
                self.render()
        
        return self.reset(), reward, self.curr_episode >= self.episodes, {}
            

    def moveTarget(self, idx):
        # Check if this target has been oncourse for max allowed time or it reached its destination
        if self.targetSteps[idx] == 0 or (abs(self.targetDestinations[idx][0] - self.targetLocations[idx][0]) < 1 and 
            abs(self.targetDestinations[idx][1] - self.targetLocations[idx][1]) < 1): #To prevent to & fro movement over destination
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


    def moveAgent(self, index, dest):
        if self.agentPosIncrements[index][0] == -1000.0 or self.agentPosIncrements[index][1] == -1000.0:
            self.agentPosIncrements[index] = self.calculateIncrements(self.agentLocations[index], dest, self.agentSpeed)

        self.agentLocations[index] += self.agentPosIncrements[index]

        if self.agentLocations[index][0] < 0:
            self.agentLocations[index][0] = 0
        if self.agentLocations[index][0] > self.gridWidth:
            self.agentLocations[index][0] = self.gridWidth
        if self.agentLocations[index][1] < 0:
            self.agentLocations[index][1] = 0
        if self.agentLocations[index][1] > self.gridHeight:
            self.agentLocations[index][1] = self.gridHeight

        #To prevent to & fro movement over destination
        if abs(dest[0] - self.agentLocations[index][0]) < 1 and abs(dest[1] - self.agentLocations[index][1]) < 1:
            return True
        else:
            return False

    
    def calculateAgentRewards(self):
        curr_reward = np.zeros(self.numAgents)
        reward_awarded_to = np.zeros(self.numTargets)

        for i, t in enumerate(self.targetLocations):
            nearestAgent = -1
            nearestdist = 2147483647.0
            for j, a in enumerate(self.agentLocations):
                distance = self.distance(a, t)
                if distance <= self.sensorRange and distance < nearestdist:
                    nearestdist = distance
                    nearestAgent = j

            if nearestAgent != -1:
                curr_reward[nearestAgent] += 1
                reward_awarded_to[i] = nearestAgent
            else:
                reward_awarded_to[i] = -1
        
        return curr_reward, reward_awarded_to

    
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

    
    def getAgentPosition(self, i):
        return self.agentLocations[i]


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

            self.agents_geom = []
            for i in self.agentLocations:
                point = (self.scale[0]*i[0] + borderOffset, self.scale[1]*i[1] + borderOffset)
                location = rendering.Transform(translation=point)
                axle = rendering.make_circle(4.0)
                axle.add_attr(location)
                axle.set_color(0.0, 0.0, 1.0)
                self.agents_geom.append(location)
                self.viewer.add_geom(axle)

                coverage = self.viewer.draw_circle(radius=self.scale[0]*self.sensorRange, res=30, filled=False)
                coverage.add_attr(location)
                coverage.set_color(0.5, 0.5, 0.8)
                self.viewer.add_geom(coverage)

        else:
            for i, t in enumerate(self.targets_geom):
                point = (self.scale[0]*self.targetLocations[i][0] + borderOffset, 
                            self.scale[1]*self.targetLocations[i][1] + borderOffset)

                t.set_translation(point[0], point[1])

            for i, a in enumerate(self.agents_geom):
                point = (self.scale[0]*self.agentLocations[i][0] + borderOffset, 
                            self.scale[1]*self.agentLocations[i][1] + borderOffset)

                a.set_translation(point[0], point[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def stopRender(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None