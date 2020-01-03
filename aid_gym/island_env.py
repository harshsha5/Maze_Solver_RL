import numpy as np

class DiscreteSpace:

    def __init__(self, size, description, min, max):
        self.size = size
        self.description = description

        self.min = np.array(min)
        self.max = np.array(max)

    def sample(self):
        return np.random.randint(0, 2, self.size)


class IslandEnv:
    MAP_SIZE = 2048
    REWARD_GOAL = 1000
    REWARD_WATER = -1000

    def __init__(self):

        with open("assets/elevation.data", "rb") as elevation_file:
            self.elevation = np.frombuffer(elevation_file.read(), dtype=np.uint8).reshape(self.MAP_SIZE,self.MAP_SIZE)
        with open("assets/overrides.data", "rb") as overrides_file:
            self.overrides = np.frombuffer(overrides_file.read(), dtype=np.uint8).reshape(self.MAP_SIZE, self.MAP_SIZE)

        self.stateSpace = DiscreteSpace(4, "[x, y, x_goal, y_goal]", np.zeros(4), np.ones(4)*IslandEnv.MAP_SIZE-1)
        self.actionSpace = DiscreteSpace(2, "", [-1, -1], [1, 1])

        self.reset()

    def reset(self):
        """

        This function will reset the robot to a random initial position, giving it a random goal on the island

        :return: the initial state and goal position
        """
        self.s = self._sampleValidPosition()
        self.s_goal = self._sampleValidPosition()
        return np.concatenate([self.s, self.s_goal])

    def render(self):
        pass

    def step(self, a):
        """

        Try to perform action a with the robot on the island.

        :param a: action to be performed. Has to be in set [0,1], [1,0], ... [1,1], [-1,-1]
        :return: [new_state, reward, done, info]
        """

        r = 0.0
        d = False

        s_ = np.clip(self.s + np.array(a), 0, IslandEnv.MAP_SIZE-1)

        dh = self._getHeightDifference(self.s, s_)
        prob = self._getSuccessProbability(dh)
        r = -np.sqrt(np.square(a).sum())

        # randomly fail with some probability
        if np.random.rand() <= prob:
            if self._isValidPosition(s_):
                if np.array_equal(s_, self.s_goal):
                    r = self.REWARD_GOAL
                    d = True
            else:
                r = self.REWARD_WATER
                d = True
            
            self.s = s_

        return np.concatenate([self.s, self.s_goal]), r, d, {}

    def _isValidPosition(self, pos):
        return not (self.overrides[pos[0], pos[1]] & 0x40)

    def _sampleValidPosition(self):
        while True:
            pos = [np.random.randint(0,self.MAP_SIZE), np.random.randint(0,self.MAP_SIZE)]
            if self._isValidPosition(pos):
                return np.array(pos)

    def _getHeightDifference(self, s1, s2):
        e1 = self.elevation[s1[0], s1[1]]
        e2 = self.elevation[s2[0], s2[1]]
        if e1 == e2:
            hd = 0.0
        else:
            distance = np.sqrt(np.square(s2 - s1).sum())
            hd = (float(e2)-e1)/distance
        return hd

    def _getSuccessProbability(self, dh):
        return 1/np.exp(np.abs(dh))





