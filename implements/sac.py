from torch import nn

class SAC:
    def __init__(
            self, 
            env, 
            policy: nn.module, 
            qf1: nn.module, 
            qf2: nn.module, 
            vf: nn.module, 
            pool_size: int,
            tau: float,
            lr: float,
            scale_reward: float,
            discount: float,
            episode_num: int,
            batch_size: int
    ):
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.pool_size = pool_size
        self.tau = tau
        self.lr = lr
        self.scale_reward = scale_reward
        self.discount = discount
        self.episode_num = episode_num
        self.batch_size = batch_size

    def _update_critic(self):
        print('update critic(q1, q2)')

    def _update_actor(self):
        print('update actor(V, pi)')

    def _smooth_target(self):
        print('smooth target(V)')

    def _evaluate(self):
        print('result!')

    def _achieve_goal(self) -> bool:
        return False

    def train(self) -> float:
        for episode in range(self.episode_num):
            # One episode
            observation, info = self.env.reset()
            done = truncated = False
            while not (done or truncated):
                action = self.env.action_space.sample()
                observation, reward, terminated, truncated, info = self.env.step(action)
            self.env.close()

            self._evalute()

            for gradient_step in range(self.batch_size):
                self._update_critic()
                self._update_actor()
                self._smooth_target()
        print('train complete')