import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck process.Ornstein–Uhlenbeck 噪点
    """

    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


def testNoise():
    noise_test = OUNoise(4)
    noise_test1 = OUNoise(4, theta=0.6, sigma=0.3)
    for _ in range(50):
        print(noise_test.sample(), ' ', noise_test1.sample())


# testNoise()
