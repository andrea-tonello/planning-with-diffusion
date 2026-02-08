import numpy as np

# Walls as line segments: (x1, y1, x2, y2)
# Using [0, 1] x [0, 1] coordinate space

MAZE_CONFIGS = {
    "umaze": {
        "walls": [
            # Outer boundary
            (0.0, 0.0, 1.0, 0.0),  # bottom
            (0.0, 1.0, 1.0, 1.0),  # top
            (0.0, 0.0, 0.0, 1.0),  # left
            (1.0, 0.0, 1.0, 1.0),  # right
            # Inner walls forming U-shape
            (0.25, 0.5, 0.75, 0.5),  # horizontal wall
            (0.5, 0.5, 0.5, 1.0),  # vertical wall up
        ],
        # Rectangular regions for sampling start/goal
        "start_region": (0.1, 0.6, 0.35, 0.9),  # top-left
        "goal_region": (0.65, 0.6, 0.9, 0.9),  # top-right
    },

    "umaze-inv": {  # Swapped start/goal. To be used during evaluation to test for generalization
        "walls": [
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
            (0.25, 0.5, 0.75, 0.5),
            (0.5, 0.5, 0.5, 1.0),
        ],
        "start_region": (0.65, 0.6, 0.9, 0.9),
        "goal_region": (0.1, 0.6, 0.35, 0.9),
    },

    "smaze": {
        "walls": [
            (0.0, 0.0, 1.0, 0.0),  # bottom
            (0.0, 1.0, 1.0, 1.0),  # top
            (0.0, 0.0, 0.0, 1.0),  # left
            (1.0, 0.0, 1.0, 1.0),  # right

            (0.30, 0.67, 1.0, 0.67),  # top line
            (0.0, 0.33, 0.70, 0.33),  # bottom line
        ],
        "start_region": (0.80, 0.67, 1.0, 1.0),
        "goal_region": (0.0, 0.0, 0.20, 0.33),
    },

    "smaze-inv": {  # Swapped start/goal. To be used during evaluation to test for generalization
        "walls": [
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),

            (0.30, 0.67, 1.0, 0.67),
            (0.0, 0.33, 0.70, 0.33),
        ],
        "start_region": (0.0, 0.0, 0.20, 0.33),
        "goal_region": (0.80, 0.67, 1.0, 1.0),
    },

    "simple": {
        # Simple open maze (no internal walls)
        "walls": [
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
        ],
        "start_region": (0.1, 0.1, 0.3, 0.3),
        "goal_region": (0.7, 0.7, 0.9, 0.9),
    },
}


class Maze2DEnv:
    """
    2D maze environment which allows for different maze layouts.

    State: (x, y) position in [0, 1] x [0, 1]
    Action: (dx, dy) velocity in [-max_action, max_action]

    The agent navigates from a start position to a goal position,
    avoiding walls. Reward is sparse: 1.0 at goal, 0.0 elsewhere.
    """

    def __init__(self, maze_type="umaze", max_action=0.05, goal_threshold=0.05, dt=1.0):
        """
        Initialize the maze environment.
        - In:
            - maze_type: type of maze layout ("umaze" / "smaze" / "simple")
            - max_action: maximum velocity magnitude
            - goal_threshold: distance threshold for goal success
            - dt: timestep for dynamics
        """
        if maze_type not in MAZE_CONFIGS:
            raise ValueError(f"Unknown maze type: {maze_type}")

        config = MAZE_CONFIGS[maze_type]
        self.walls = config["walls"]
        self.start_region = config["start_region"]
        self.goal_region = config["goal_region"]

        self.state_dim = 2
        self.action_dim = 2
        self.max_action = max_action
        self.goal_threshold = goal_threshold
        self.dt = dt

        self.state = None
        self.goal = None
        self._rng = np.random.default_rng()

    def seed(self, seed):
        self._rng = np.random.default_rng(seed)

    def reset(self, start=None, goal=None):
        """Reset the environment to an initial state."""
        if start is not None:
            self.state = np.array(start, dtype=np.float32)
        else:
            self.state = self._sample_from_region(self.start_region)

        if goal is not None:
            self.goal = np.array(goal, dtype=np.float32)
        else:
            self.goal = self._sample_from_region(self.goal_region)

        return self.state.copy()

    def step(self, action):
        """
        Execute an action (actions are determined by velocities dx, dy)
        -In:
            - action: (dx, dy) velocity
        - Out:
            - next_state, reward, done, info
        """
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -self.max_action, self.max_action)

        # Compute next state
        next_state = self.state + action * self.dt

        # Check wall collision using line intersection
        if self._collides_with_wall(self.state, next_state):
            next_state = self.state  # Stay in place

        # Clip to bounds
        next_state = np.clip(next_state, 0.01, 0.99)

        self.state = next_state

        # Sparse reward
        dist_to_goal = np.linalg.norm(self.state - self.goal)
        done = dist_to_goal < self.goal_threshold
        reward = 1.0 if done else 0.0

        return self.state.copy(), reward, done, {"distance": dist_to_goal}

    def sample_goal(self):
        return self._sample_from_region(self.goal_region)

    def get_goal(self):
        return self.goal.copy()

    def _sample_from_region(self, region):
        """Sample a valid point from a rectangular region."""
        x_min, y_min, x_max, y_max = region
        for _ in range(100):  # Try up to 100 times
            x = self._rng.uniform(x_min, x_max)
            y = self._rng.uniform(y_min, y_max)
            point = np.array([x, y], dtype=np.float32)
            if self._is_valid_position(point):
                return point
        # Fallback: return center of region
        return np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)

    def _is_valid_position(self, pos):
        """Check if given position is not inside a wall."""
        # For line-segment walls, we check if position is valid (i.e. not too close to walls)
        margin = 0.02
        x, y = pos
        if x < margin or x > 1 - margin or y < margin or y > 1 - margin:
            return False
        return True

    def _collides_with_wall(self, start, end):
        """Check if movement from start to end crosses a wall."""
        for wall in self.walls:
            if self._line_segments_intersect(start, end, wall):
                return True
        return False

    def _line_segments_intersect(self, p1, p2, wall):
        """Check if line segment (p1, p2) intersects wall segment."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3, x4, y4 = wall

        # Using cross product method for line intersection
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        p3 = np.array([x3, y3])
        p4 = np.array([x4, y4])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
            (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
        ):
            return True

        return False

    def get_walls(self):
        # For visualization purposes
        return self.walls

    def render(self):
        # Render the environment (placeholder for visualization)
        pass
