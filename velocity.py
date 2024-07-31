from config import Config
import numpy as np
import math
import time


class VelocityModel:
    def __init__(self, x0, x1):
        self.start_t = time.time()
        self.x0 = x0
        self.x1 = x1
        self.a = Config.ACCELERATION / Config.DISPLAY_CELL_SIZE
        self.d = Config.DECELERATION / Config.DISPLAY_CELL_SIZE
        self.v_max = Config.MAX_SPEED / Config.DISPLAY_CELL_SIZE
        self.v_mid = self.v_max
        self.total_time = 0
        self.a_time = 0
        self.d_time = 0
        self.v_time = 0
        self.dist = np.linalg.norm(self.x1 - self.x0)
        self.a_vec = None
        self.d_vec = None
        self.v_vec = None
        self.w1 = None
        self.w2 = None

    def solve(self):

        time_a = self.v_max / self.a
        time_d = self.v_max / self.d

        dist_a = time_a * self.v_max / 2
        dist_d = time_d * self.v_max / 2

        if dist_a + dist_d > self.dist:
            self.v_mid = math.sqrt(2 * self.dist * self.a * self.d / (self.a + self.d))
            self.a_time = self.v_mid / self.a
            self.d_time = self.v_mid / self.d
        else:
            self.a_time = time_a
            self.d_time = time_d
            self.v_time = (self.dist - dist_a - dist_d) / self.v_max

        self.total_time = self.a_time + self.v_time + self.d_time

        v_norm_vec = self.x1 - self.x0
        if self.dist > 0.0:
            v_norm_vec /= self.dist
        self.a_vec = self.a * v_norm_vec
        self.d_vec = -self.d * v_norm_vec
        self.v_vec = self.v_mid * v_norm_vec

        self.w1 = self.x0 + .5 * self.a_vec * self.a_time ** 2
        self.w2 = self.w1 + self.v_vec * self.v_time

    def get_location(self, t):
        dt = t - self.start_t
        if dt < 0:
            return self.x0
        elif dt < self.a_time:
            return self.x0 + .5 * self.a_vec * dt ** 2
        elif dt < self.a_time + self.v_time:
            return self.w1 + self.v_vec * (dt - self.a_time)
        elif dt < self.total_time:
            ddt = dt - self.a_time - self.v_time
            return self.w2 + .5 * self.d_vec * ddt ** 2 + self.v_vec * ddt
        else:
            return self.x1


if __name__ == '__main__':
    t = time.time()
    vm = VelocityModel(np.array([.0, .0, .0]), np.array([.0, .0, .0]))
    vm.solve()
    print(vm.get_location(t))
    print(vm.get_location(t+1))
    print(vm.get_location(t+2))
    print(vm.get_location(t+3))
    print(vm.get_location(t+4))
    print(vm.get_location(t+5))
    print(vm.get_location(t+6))
    print(vm.get_location(t+7))
    print(vm.get_location(t+vm.total_time))
    print(vm.total_time)
