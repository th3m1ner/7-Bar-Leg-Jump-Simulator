import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import pandas as pd
import os


# Easy to use and organzied robot config
class RobotConfig:

    def __init__(self):

        # Motor and mass
        self.motor_torque = 10
        self.robot_mass = 2

        # Const torque spring
        self.spring_const_torque = 0

        # general torsion spring
        self.k1 = 0  # N/m
        self.q1bar = np.pi + np.radians(100)  # rest angle of spring

        # General spring properties
        # Spring mcmaster pn = 5108N79
        # Used for example
        self.spring_gen_l1_attach = 0.15
        self.spring_gen_fixed = np.array([0.05, -0.1])
        self.spring_gen_k = 10 * 175.126835  #lbf / in to N/m is 1:175.126835
        self.spring_gen_base = 6 * 0.0254  # in to m is 1:0.0254
        self.spring_gen_max_load = 54.3 * 4.44822  #lbf to N is 1:4.44822
        self.spring_gen_max_len = 10.97 * 0.0254  # in to m is 1:0.0254
        # offset for spring for less constrained placement
        self.spring_gen_ang_offset = np.radians(0)

        # Properties of the links
        self.link_l1 = 259.64
        self.link_l2 = 287

        # Angle Limits
        self.angle_min = np.radians(179)
        self.angle_max = np.radians(240)

        # Virtual Link propeties
        self.vL1 = 225
        self.vL2 = 100
        self.vL3 = 36
        self.th1_offset = np.radians(2.323)
        self.link_offset_angle = np.radians(17)

        #d/dt delta
        self.d = 1e-8

        # Mass Properties
        self.m1 = 0.158
        self.m2 = 0.184
        self.wheel_m = 0.297
        self.motor_m = 0.393
        self.battery_m = 0.311

        # Physical properties
        self.a1 = self.link_l1 * 0.001
        self.a2 = self.link_l2 * 0.001
        self.k2 = 0  # N/m
        self.q2bar = 0
        self.ell1 = self.a1 / 2
        self.ell2 = self.a2 / 2
        self.w1 = 0.05
        self.w2 = 0.05
        self.Iz1 = (1 / 12) * self.m1 * (self.a1**2 + self.w1**2)
        self.Iz2 = (1 / 12) * self.m2 * (self.a2**2 + self.w2**2)
        self.damping = 0.0
        self.grav = 9.81
        self.external_forces = [0, 0]
        self.eff_payload_mass = 0

        # Wheel Properties modeled as a solid disk
        self.wheel_radius = 0.1
        self.wheel_mass = self.wheel_m + self.motor_m  # simplifying the motor and wheel together
        self.I_wheel = 0.5 * self.wheel_mass * self.wheel_radius**2

        # Robot Body Properties
        self.robot_a = 0.1
        self.robot_w = 0.2
        self.I_robot = (1 / 12) * self.robot_mass * (self.robot_a**2 +
                                                     self.robot_w**2)

        # Total weight and mass
        self.m_total = self.m1 + self.m2 + self.robot_mass + self.wheel_mass
        self.weight = self.m_total * self.grav


# Store the data for additional plotting
class SimulationData:

    def __init__(self, tmax=1.5, dt=0.001):
        self.tmax = tmax
        self.dt = dt
        self.n = n = int(tmax / self.dt)

        self.th_traj = np.zeros((3, n))  # th1, dth1, th2
        self.y_traj = np.zeros((2, n))  # y, dy
        self.com_vel = np.zeros((2, n))  # x y com vel
        self.X = np.zeros((3, n))
        self.Y = np.zeros((3, n))
        self.com_x = np.zeros((2, n))
        self.com_y = np.zeros((2, n))
        self.grounded_state = np.zeros(n, dtype=bool)
        self.phys_x = np.zeros((16, n))
        self.phys_y = np.zeros((16, n))
        self.jump_y_estimate = 0
        self.takeofff_vel = 0
        self.time = np.zeros(n)

    # Exporter to csv for easier plotting
    def export_to_csv(self, filename="results.csv"):

        headers = [
            'Time s',
            'th1 rad',
            'dth1 rad/s',
            'th2 rad',
            'y_pos m',
            'dy_vel m/s',
            'com_vel x m/s',
            'com vel y m/s',
            'com x pos_1',
            'com x pos_2',
            'com y pos_1',
            'com y pos_2',
            'Grounded State bool',
        ]

        data = (
            self.time[np.newaxis, :],
            self.th_traj,
            self.y_traj,
            self.com_vel,
            self.com_x,
            self.com_y,
            self.grounded_state[np.newaxis, :].astype(int),
        )

        full_data_matrix = np.vstack(data)

        data = full_data_matrix.T

        try:
            df = pd.DataFrame(data, columns=headers)

            df['Jump Height Estimate mm'] = self.jump_y_estimate
            df['Takeoff Velocity mp/s'] = self.takeofff_vel

            df.to_csv(filename, index=False)
            print(f"Successfully exported to {filename}")

        except Exception as e:
            print(f"Error: {e}")


# Circle intersection helper function
def circle_intersection(p1, r1, p2, r2):
    # Return the two intersection points between two circles
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    d = np.linalg.norm(p2 - p1)

    if d == 0 or d > r1 + r2 or d < np.abs(r1 - r2):
        return None, None

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - a**2
    if h_sq < 0:
        return None, None

    h = np.sqrt(h_sq)
    mid = p1 + a * (p2 - p1) / d
    off = h * np.array([-(p2[1] - p1[1]) / d, (p2[0] - p1[0]) / d])

    return mid + off, mid - off


# General function for plotting simulation data
def plot_data(x, y, title="Plot", xlabel="X", ylabel="Y"):

    # Remove the last element
    # since we are defining th1 and dth1 together, the last row is left blank
    # and is confusing when plotting

    x = x[:-1]
    y = y[:-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# Helper for easier link plotting
class Link2D:

    def __init__(
        self,
        length,
        angle=0.0,
        parent=None,
        origin=None,
        mass=0,
        com_ratio=0.5,
    ):
        self.length = float(length)
        self.angle = float(angle)
        self.parent = parent
        self.origin = np.array(origin if parent is None else parent.get_end(),
                               dtype=float)
        self.mass = mass
        self.com_ratio = com_ratio

    def get_rotation_matrix(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([[c, -s], [s, c]])

    def get_global_rotation(self):
        if self.parent is None:
            return self.get_rotation_matrix()
        return self.parent.get_global_rotation() @ self.get_rotation_matrix()

    def get_start(self):
        return self.origin if self.parent is None else self.parent.get_end()

    def get_end(self):
        return self.get_start() + self.get_global_rotation() @ np.array(
            [self.length, 0.0])

    def update_angle(self, angle):
        self.angle = float(angle)

    def get_com_pt(self):
        start = self.get_start()
        end = self.get_end()
        return start + self.com_ratio * (end - start)

    def translate(self, delta: np.array):
        self.origin = self.origin.astype(float)
        if self.parent is None:
            self.origin += delta
        else:
            self.origin = self.get_start() + delta


# Main robot leg class
class Robot:

    def __init__(self):
        self.c = RobotConfig()

        # States
        self.state_ground = True
        self.state_jump = False
        self.state_landing = False
        self.state_landed = False

        # Setup link relations
        self.link1 = Link2D(length=self.c.link_l1, origin=[0, 0])
        self.link2 = Link2D(length=self.c.link_l2, parent=self.link1)

        self.vel_correction = 0

    def get_plotting_xys(self, th1, th2=None):

        # Calc th2 if not passed in
        if (th2 == None):
            th2 = self.calc_th2(th1)

        # update links
        self.link1.update_angle(th1)
        self.link2.update_angle(th2)

        if (self.state_ground):
            # Translate to center on p7 if on the ground
            p7 = self.link2.get_end()
            self.link1.translate(-p7)

        # get points for plotting
        p0 = self.link1.get_start()
        p1 = self.link1.get_end()
        p2 = self.link2.get_end()

        # pack for plotting links
        X = (p0[0], p1[0], p2[0])
        Y = (p0[1], p1[1], p2[1])

        # centers of mass (also shifted)
        com_pt1 = self.link1.get_com_pt()
        com_pt2 = self.link2.get_com_pt()

        # com for plotting
        com_x = [com_pt1[0], com_pt2[0]]
        com_y = [com_pt1[1], com_pt2[1]]

        return X, Y, com_x, com_y

    # Function to overlay physical system over virtual system
    def get_7_bar_plotting_xys(self, th1):

        th1 = th1 + np.pi - self.c.th1_offset

        t2 = th1 + self.c.link_offset_angle
        l1 = self.c.vL1
        l2 = self.c.vL2
        l3 = self.c.vL3
        l4 = self.c.link_l2

        p1 = [0, 0]
        p2 = [0, l2]
        p3 = [-l1 * np.cos(th1), -l1 * np.sin(th1) + l2]
        p4 = [
            -l1 * np.cos(th1) - l3 * np.cos(t2),
            l2 - l1 * np.sin(th1) - l3 * np.sin(t2)
        ]

        p5 = [-l1 * np.cos(th1), -l1 * np.sin(th1)]

        pa, pb = circle_intersection(p4, l4, p5, l1)
        if pa is None:
            return 0.0

        p7 = pa

        p6 = [p7[0], p7[1] + l2]

        x = [p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0]]
        y = [p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1]]

        x_plot = [
            x[0], x[1], x[2], x[3], x[2], x[4], x[0], x[4], x[2], x[3], x[6],
            x[6], x[5], x[2], x[4], x[6]
        ]
        y_plot = [
            y[0], y[1], y[2], y[3], y[2], y[4], y[0], y[4], y[2], y[3], y[6],
            y[6] + l2 / 2, y[5], y[2], y[4], y[6]
        ]

        return x_plot, y_plot

    def get_spring_plotting_xys(self, q1):
        P0 = self.c.spring_gen_fixed * 1000
        l = self.c.spring_gen_l1_attach * 1000
        P1 = np.array([
            l * np.cos(q1 + self.c.spring_gen_ang_offset),
            l * np.sin(q1 + self.c.spring_gen_ang_offset)
        ])

        P2 = np.array([l * np.cos(q1), l * np.sin(q1)])

        X = [P0[0], P1[0], P2[0], 0]
        Y = [P0[1], P1[1], P2[1], 0]

        return X, Y

    def calc_J_com_1(self, q1):
        return np.array([(-self.c.ell1 * np.sin(q1)),
                         (self.c.ell1 * np.cos(q1))])

    def calc_J_com_2(self, q1, q2):
        # since we are using this as effectly a 1dof system we need to
        # combine in the second coulm of the J Matrix
        # Chain rule as it is only dependant on q1
        # v = J * dq = [J11 J12 ; J21 J22] * [dq1 dq2]
        # vx = J11*dq1 + J12 * dq2
        # vy = J21*dq1 + J22 * dq2
        # since dq2 = dq2/dq1*dq1

        dq2_dq1 = self.calc_dtheta2(q1)
        J11 = -self.c.ell2 * np.sin(q1 + q2) - self.c.a1 * np.sin(q1)
        J12 = -self.c.ell2 * np.sin(q1 + q2) * dq2_dq1

        J21 = self.c.ell2 * np.cos(q1 + q2) + self.c.a1 * np.cos(q1)
        J22 = self.c.ell2 * np.cos(q1 + q2) * dq2_dq1

        J = np.array([J11 + J12, J21 + J22])

        return J

    def calc_J_eff(self, q1, q2):
        # since we are using this as effectly a 1dof system we need to
        # combine in the second coulm of the J Matrix
        # Chain rule as it is only dependant on q1
        # v = J * dq = [J11 J12 ; J21 J22] * [dq1 dq2]
        # vx = J11*dq1 + J12 * dq2_dq1
        # vy = J21*dq1 + J22 * dq2_dq1
        # since dq2 = dq2_dq1*dq1

        dq2_dq1 = self.calc_dtheta2(q1)
        J11 = -self.c.a2 * np.sin(q1 + q2) - self.c.a1 * np.sin(q1)
        J12 = -self.c.a2 * np.sin(q1 + q2) * dq2_dq1

        J21 = self.c.a2 * np.cos(q1 + q2) + self.c.a1 * np.cos(q1)
        J22 = self.c.a2 * np.cos(q1 + q2) * dq2_dq1

        J = np.array([J11 + J12, J21 + J22])

        return J

    def calc_th2(self, th1):

        p4 = np.array(
            [self.c.link_l1 * np.cos(th1), self.c.link_l1 * np.sin(th1)])
        p5 = np.array([
            self.c.vL1 * np.cos(th1 - self.c.th1_offset),
            self.c.vL1 * np.sin(th1 - self.c.th1_offset) - self.c.vL2
        ])

        pa, pb = circle_intersection(p4, self.c.link_l2, p5, self.c.vL1)
        if pa is None:
            return 0.0

        p7 = pa
        vec = p7 - p4
        th2 = np.arctan2(-vec[1], vec[0])
        th2 = np.pi - (th2 + (th1 - np.pi))
        th2 = th2
        return th2

    def calc_dtheta2(self, th1):
        d = self.c.d
        return (self.calc_th2(th1 + d) - self.calc_th2(th1 - d)) / (2 * d)

    def calc_ddtheta2(self, th1):
        d = self.c.d
        return (self.calc_dtheta2(th1 + d) - self.calc_dtheta2(th1 - d)) / (2 *
                                                                            d)

    def calc_mass_matrix(self, q1, q2, dq2_dq1):
        # Pull out the constants from the config for ease of use
        a1, a2 = self.c.a1, self.c.a2
        ell1, ell2 = self.c.ell1, self.c.ell2
        m1, m2 = self.c.m1, self.c.m2
        Iz1, Iz2 = self.c.Iz1, self.c.Iz2
        m3, Iz3 = self.c.wheel_mass, self.c.I_wheel
        m4, Iz4 = self.c.robot_mass, self.c.I_robot
        grounded = self.state_ground

        # Calc eff inerta position
        r_eff = np.array([(a2 * np.cos(q1 + q2) + a1 * np.cos(q1)),
                          (a2 * np.sin(q1 + q2) + a1 * np.sin(q1))])

        r_eff_sq = np.dot(r_eff, r_eff)
        M_wheel = Iz3 * (1 + dq2_dq1)**2 + m3 * r_eff_sq
        M_robot = Iz4 * (1 + dq2_dq1)**2 + m4 * r_eff_sq

        M11 = Iz1 + Iz2 + m1 * ell1**2 + m2 * (a1**2 + ell2**2 +
                                               2 * a1 * ell2 * np.cos(q2))

        M12 = Iz2 + m2 * (ell2**2 + a1 * ell2 * np.cos(q2))

        M22 = Iz2 + m2 * ell2**2

        # Chain rule to for 1dof system
        M = M11 + 2 * M12 * dq2_dq1 + M22 * (dq2_dq1**2)

        # If the robot is grounded only add the body mass, if in the air add both body and wheel
        if (grounded):
            M = M + M_robot
        else:
            M = M + M_wheel + M_robot

        return M

    def calc_coriolis_terms(self, dq1, q2, dq2_dq1):
        # Pull out the constants from the config for ease of use
        a1, a2 = self.c.a1, self.c.a2
        ell1, ell2 = self.c.ell1, self.c.ell2
        m1, m2 = self.c.m1, self.c.m2
        m3 = self.c.wheel_mass
        m4 = self.c.robot_mass
        grounded = self.state_ground

        # only robot mass in the grounded state, and both in air state
        if (grounded):
            h = (m2 * a1 * ell2 + m4 * a1 * a2) * np.sin(q2)
        else:
            h = (m2 * a1 * ell2 + m3 * a1 * a2 + m4 * a1 * a2) * np.sin(q2)

        C = -h * dq2_dq1 * (2 + dq2_dq1) * (dq1**2)
        return C

    def calc_spring_torques(self, q1):
        k1, q1bar = self.c.k1, self.c.q1bar

        # general torsion spring
        Fs = k1 * (q1 - q1bar)

        # constant torque spring
        Fcs = self.c.spring_const_torque

        # General Spring
        l_attach = self.c.spring_gen_l1_attach
        P0 = self.c.spring_gen_fixed
        q1_offst = self.c.spring_gen_ang_offset
        r = np.array([
            l_attach * np.cos(q1 + q1_offst), l_attach * np.sin(q1 + q1_offst)
        ])

        # Displacement of the spring
        dis_vec = r - P0

        # Current spring length ||V_F||
        current_len = np.linalg.norm(dis_vec)

        # Spring Force
        F_spring = self.c.spring_gen_k * (current_len - self.c.spring_gen_base)

        # Vectorize the force
        F = F_spring * (dis_vec / current_len)

        # Error check if we break the spring
        if (current_len > self.c.spring_gen_max_len):
            print(
                f"Error: spring length exceeds max rated length: {current_len} > {self.c.spring_gen_max_len} when at q1 = {np.degrees(q1)}"
            )

        if (F_spring > self.c.spring_gen_max_load):
            print(
                f"Error: spring length exceeds max rated load: {F_spring} > {self.c.spring_gen_max_load} when at q1 = {np.degrees(q1)}"
            )

        # T_extend = r cross F tp spve for the torque around q1
        T_extend = np.cross(r, F)

        return Fs, Fcs, T_extend

    def calc_gravity_torques(self, q1, q2):
        # Pull out the constants from the config for ease of use
        a1, a2 = self.c.a1, self.c.a2
        ell1, ell2 = self.c.ell1, self.c.ell2
        m1, m2 = self.c.m1, self.c.m2
        grav = self.c.grav
        m3 = self.c.wheel_mass
        m4 = self.c.robot_mass
        grounded = self.state_ground

        # Calc eff inerta position
        r_eff = np.array([(a2 * np.cos(q1 + q2) + a1 * np.cos(q1)),
                          (a2 * np.sin(q1 + q2) + a1 * np.sin(q1))])

        #calc com of link 1 and 2
        com_1 = np.array([ell1 * np.cos(q1), ell1 * np.sin(q1)])
        com_2 = np.array([(ell2 * np.cos(q1 + q2) + a1 * np.cos(q1)),
                          (ell2 * np.sin(q1 + q2) + a1 * np.sin(q1))])

        g_l1 = m1 * grav * com_1[0]
        g_l2 = m2 * grav * com_2[0]
        g_wheel = m3 * grav * r_eff[0]
        g_robot = m4 * grav * r_eff[0]

        # add the robot if on the ground and add the wheel + robot if in the air
        if (grounded):
            G = g_l1 + g_l2 + g_robot
        else:
            G = g_l1 + g_l2 + g_wheel + g_robot

        return G

    def calc_com_velocity(self, q1, dq1, q2, dq2):
        m1, m2 = self.c.m1, self.c.m2
        m3 = self.c.wheel_mass
        m4 = self.c.robot_mass
        m_total = self.c.m_total

        # -------------------------- System COM vertical velocity --------------------------
        # need to calculate hte vertical velocity inorder to calc vertical jump displacment

        # COM_y * q_dot ~ v = r * omega
        # v1 = com_1 * dq1
        # v2 = com_2 * (dq1 + dq2)
        # v3 = 0
        # v4 = r_eff * (dq1 + dq2)

        # v = J * omega
        v1 = self.calc_J_com_1(q1) * dq1
        v2 = self.calc_J_com_2(q1, q2) * (dq1 + dq2)
        v3 = 0
        v4 = self.calc_J_eff(q1, q2) * (dq1 + dq2)

        # Total COM velocity
        # v_com_y = sum of all masses * sum of vel / total mass
        com_vel = (m1 * v1 + m2 * v2 + m3 * v3 + m4 * v4) / m_total
        # self.com_vel = [com_vel[1], com_vel[0]]

        return com_vel

    def calc_dynamics(self, q1, dq1):
        # Pull out the constants from the config for ease of use
        damping = self.c.damping
        grav = self.c.grav
        m3 = self.c.wheel_mass
        m4 = self.c.robot_mass
        grounded = self.state_ground
        th1_max, th1_min = self.c.angle_max, self.c.angle_min

        # -------------------------- Theta 2 calcs --------------------------
        q2 = self.calc_th2(q1)
        # Chain rule dq2/dq1
        dq2_dq1 = self.calc_dtheta2(q1)
        dq2 = dq2_dq1 * dq1

        # -------------------------- Mass location helpers --------------------------

        # -------------------------- Mass Matrix --------------------------

        M = self.calc_mass_matrix(q1, q2, dq2_dq1)

        # -------------------------- Coriolis terms --------------------------

        C = self.calc_coriolis_terms(dq1, q2, dq2_dq1)

        # -------------------------- Spring force --------------------------
        Fs, Fcs, T_extend = self.calc_spring_torques(q1)

        # -------------------------- Dampening terms --------------------------

        Fd = damping * dq1

        # -------------------------- Gravitational Torques --------------------------

        G = self.calc_gravity_torques(q1, q2)

        # -------------------------- Point Loads --------------------------
        # these are the effective loads of the body and wheel on the End effector
        # Jacobian (generalzied 1dof, pendulum): J = [dx2/dth1; dy2/dth1]

        J = self.calc_J_eff(q1, q2)

        Fwhl = J @ np.array([0, m3 * grav])
        Fbdy = J @ np.array([0, m4 * grav])

        # -------------------------- Motor Torque --------------------------
        # Applied motor torque is simply tau
        tau = self.c.motor_torque

        # -------------------------- System COM vertical velocity --------------------------

        self.com_vel = self.calc_com_velocity(q1, dq1, q2, dq2)

        # -------------------------- ddq1 --------------------------

        # If it is on the ground, we will reverse the gravity and translate the body above
        # the x axis to show proper motion
        if (grounded):
            ddq1 = 1 / M * (-C - Fs - T_extend - Fd + G + Fbdy + tau + Fcs)
        else:
            ddq1 = 1 / M * (-C - Fs - T_extend - Fd - G - Fwhl - tau - Fcs)

        # -------------------------- Limit Evaluation --------------------------
        # When we hit the max angles, the system hits a phiscal hardstop, we need to evalaute
        # the force at that point

        # Limit check
        at_max_limit = (q1 >= th1_max)
        at_min_limit = (q1 <= th1_min)

        # Constraint torque (Lagrange multiplier)
        if at_max_limit and grounded:
            # An equal and opposite torque is applied to the system when it hits the hardstop
            # ddq1 = 1/M * (-C - Fs - ...) so a hardstop is -torque_applied
            # This will be used to help with robot takeoff
            # can also write it as M * ddq1 = τ + λ where λ is the lagrange multiplier
            # so when ddq1 = 0 =  τ + λ => λ = - τ

            # when we hit our max angle, continute using hte velocity and switch to jump state

            self.vel_correction = -dq1
            ddq1 = 0
            self.state_ground = False

        elif (at_max_limit and not grounded):
            self.vel_correction = -dq1
        else:
            self.state_ground = True

        if at_min_limit:
            self.vel_correction = -dq1

        return ddq1


# Simulation Loop
def simulate(robot: Robot,
             th1_0,
             dth1_0,
             debug_info=True,
             tmax=1.5,
             dt=0.001) -> SimulationData:
    if (debug_info):
        print("----- Simulation Starting -----")

    data = SimulationData(tmax, dt)

    data.th_traj[:, 0] = [th1_0, dth1_0, 0]
    y = 0

    for t in range(data.n - 1):
        data.time[t] = t

        th1 = data.th_traj[0, t]
        dth1 = data.th_traj[1, t]

        ddth1 = robot.calc_dynamics(th1, dth1)

        data.th_traj[0, t + 1] = th1 + data.dt * dth1
        data.th_traj[1, t + 1] = dth1 + data.dt * ddth1

        # concider the robot a projectile while in the air to calc the jump height
        if (robot.state_ground and not robot.state_landed):
            y = data.y_traj[0, t] = 0
            dy = data.y_traj[1, t] = robot.com_vel[1]

        # State when it hits the ground but still has velocity, used to transfer that back to the small impact on the leg
        elif (robot.state_landing):
            y = data.y_traj[0, t] = 0
            dy = data.y_traj[1, t]

            robot.state_landing = False
            robot.state_landed = True

            #calc dy for dth for the landing impact absorption, v = J_eff * omega so omega = v / j_eff
            J = robot.calc_J_eff(th1, th2)
            impact_impulse_dth = -dy / J[1]
            th1 = data.th_traj[0, t]
            data.th_traj[0, t + 1] = th1 + data.dt * impact_impulse_dth
            data.th_traj[1, t + 1] = impact_impulse_dth

        elif (robot.state_landed):
            y = data.y_traj[0, t] = 0

        else:
            # jump dynamics, simple projectile
            if not robot.state_jump:
                data.y_traj[0, t] = 0
                data.takeofff_vel = robot.com_vel[1]
                data.y_traj[1, t] = data.takeofff_vel
                robot.state_jump = True

            # ydd = (F - m*g) / m but since we launched F = 0 and then m cancels out leaving only -g
            # so only accel on it is gravity
            ddy = -robot.c.grav
            y = data.y_traj[0, t]
            dy = data.y_traj[1, t]

            data.y_traj[0, t + 1] = y + data.dt * dy
            data.y_traj[1, t + 1] = dy + data.dt * ddy

            # If at ground stop moving
            if (y < 0):
                robot.state_ground = True
                robot.state_landing = True

        # Apply velocity correction if at hard stop to sim reaction to limits
        if robot.vel_correction != 0:
            data.th_traj[1, t + 1] += robot.vel_correction
            robot.vel_correction = 0

        # Store sim info
        data.grounded_state[t] = robot.state_ground
        data.th_traj[2, t] = th2 = robot.calc_th2(th1)
        rX, rY, r_cx, r_cy = robot.get_plotting_xys(th1, th2)
        data.X[:, t] = rX
        data.Y[:, t] = rY
        data.com_x[:, t] = r_cx
        data.com_y[:, t] = r_cy
        data.phys_x[:, t], data.phys_y[:,
                                       t] = robot.get_7_bar_plotting_xys(th1)
        data.com_vel[:, t] = robot.com_vel

        # Estimated jump height for comparison
    data.jump_y_estimate = (0.5 * data.takeofff_vel**2 / robot.c.grav) * 1000

    if (debug_info):
        print("----- Simulation Complete -----")
        print(
            "----- Output -----\n"
            f"com_velocity_x:    max = {max(data.com_vel[0]):.2f}   min = {min(data.com_vel[0]):.2f} m/s\n"
            f"com_velocity_y:    max = {max(data.com_vel[1]):.2f}   min = {min(data.com_vel[1]):.2f} m/s\n"
            f"estimated jump height  = {(data.jump_y_estimate):.2f} mm\n"
            f"y_traj           max y = {(np.max(data.y_traj[0,:])*1000):.2f} mm\n"
            f"y_traj          max dy = {np.max(data.y_traj[1,:]):.2f} m/s\n"
            f"th_traj        max th1 = {(np.max(data.th_traj[0,:])):.2f} radians\n"
            f"th_traj       max dth1 = {np.max(data.th_traj[1,:]):.2f} radians/s\n"
            f"weight                 = {(robot.c.weight):.2f} N\n"
            f"jump simulated         = {robot.state_jump}")

    return data


# Function to animate the stored simulation
def animate_simulation(robot: Robot, data: SimulationData):
    # 1. Run Simulation
    fig, (ax_sim, ax_info) = plt.subplots(1,
                                          2,
                                          figsize=(9, 6),
                                          gridspec_kw={'width_ratios': [2, 1]})

    # --- LEFT AXIS: Simulation ---
    ax_sim.set_xlim(-300, 300)
    ax_sim.set_ylim(0, 1000)
    ax_sim.set_aspect("equal")
    ax_sim.grid(True)
    ax_sim.set_title("Robot Simulation")
    ax_sim.set_xlabel("E1 Axis (mm)")
    ax_sim.set_ylabel("E2 Axis (mm)")

    body_rec = np.array([robot.c.robot_w * 1000, robot.c.robot_a * 1000])
    robot_body = Rectangle((0, 0),
                           body_rec[0],
                           body_rec[1],
                           fill=True,
                           color='purple',
                           linewidth=2)
    ax_sim.add_patch(robot_body)

    wheel = Circle((0, 0),
                   radius=robot.c.wheel_radius * 1000,
                   color='black',
                   fill=True)
    ax_sim.add_patch(wheel)

    virtual_leg, = ax_sim.plot([], [], 'o-', lw=3)
    physical_leg, = ax_sim.plot([], [], 'o-', lw=3)
    foot_path, = ax_sim.plot([], [], 'r-', lw=2)
    spring, = ax_sim.plot([], [], 'o-', lw=2)

    history_x, history_y = [], []

    # --- RIGHT AXIS: Info ---
    ax_info.set_ylim(0, 1)
    ax_info.set_xlim(0, 1)
    ax_info.axis('off')
    ax_info.set_title("Live Data")

    info_text = ax_info.text(0.05,
                             0.95,
                             '',
                             transform=ax_info.transAxes,
                             ha='left',
                             va='top',
                             fontsize=10,
                             family='monospace')

    # --- Plot update ---
    def update(i):
        th1 = data.th_traj[0, i]
        X, Y = data.X[:, i], data.Y[:, i]
        y = data.y_traj[:, i]

        # Translate the y data by the height of the wheel
        plt_y = Y + wheel.radius / 2 + y[0] * 1000

        # No jump vis
        # plt_y = Y + wheel.radius / 2

        virtual_leg.set_data(X, plt_y)

        px, py = data.phys_x[:, i], data.phys_y[:, i]
        py = py + plt_y[0] - 100
        px = px + X[0]

        physical_leg.set_data(px, py)

        spring_x, spring_y = robot.get_spring_plotting_xys(th1)
        spring_y = spring_y + plt_y[0]
        spring_x = spring_x + X[0]
        spring_y = spring_y

        # do not render the spring if not used i.e k = 0
        if (robot.c.spring_gen_k != 0):
            spring.set_data(spring_x, spring_y)

        # Track body path
        if (i > 1):
            history_x.append(X[0])
            history_y.append(plt_y[0])

        foot_path.set_data(history_x, history_y)

        w_pt = [X[2], plt_y[2] + wheel.radius / 2]
        body_pt = X[0], plt_y[0] - 50
        wheel.set_center(w_pt)
        robot_body.set_xy((body_pt - (body_rec / 2)))

        status_str = (
            f"θ1              = {np.degrees(th1):.2f}°\n"
            f"dθ1             = {np.degrees(data.th_traj[1, i]):.2f}°/s\n"
            f"jump height     = {(y[0] * 1000):.2f}mm\n"
            f"vel in y jump   = {(y[1]):.2f} m/s\n"
            f"est jump h      = {data.jump_y_estimate:.2f}mm\n"
            f"grounded        = {data.grounded_state[i]}\n"
            f"weight          = {(robot.c.weight):.2f} N\n"
            f"frame           = {i}")

        info_text.set_text(status_str)

        return physical_leg, spring, virtual_leg, foot_path, wheel, robot_body, info_text

    ani = FuncAnimation(fig,
                        update,
                        frames=range(0, data.n, 2),
                        interval=0,
                        blit=True)

    plt.tight_layout()
    plt.show()


# determine what motorb to use
def plot_motor_torque_vs_height():
    count = 100
    motor_torques = range(0, count)
    jump_heights = np.zeros(count)

    for tau in motor_torques:
        robot = Robot()
        robot.c.motor_torque = tau
        data = simulate(robot,
                        th1_0=np.pi + 0.1,
                        dth1_0=0,
                        tmax=0.5,
                        debug_info=False)
        jump_heights[tau] = np.max(np.max(data.y_traj[0, :]) * 1000)
        print(f"tau = {tau} h = {jump_heights[tau]:.2f} mm")
        # animate_simulation(robot, data)

    plot_data(motor_torques,
              jump_heights,
              title="Motor Selection",
              xlabel="Motor Torque (Nm)",
              ylabel="Height (mm)")


# determine what spring value balances out the weight
def plot_spring_vs_max_th1():
    count = 10
    spring_taus = range(0, count)
    th1s = np.zeros(count)

    for tau in spring_taus:
        robot = Robot()
        robot.c.motor_torque = 0
        robot.c.k1 = tau
        data = simulate(robot,
                        th1_0=np.pi + 0.1,
                        dth1_0=0,
                        tmax=0.5,
                        debug_info=False)
        th1s[tau] = np.max(np.max(data.th_traj[0, :]))
        print(f"tau = {tau} th1 = {np.degrees(th1s[tau]):.2f} degrees")
        # animate_simulation(robot, data)
    th1s = th1s * 57.2958  #B convert to degrees

    plot_data(spring_taus,
              th1s,
              title="Spring Selection",
              xlabel="Spring Torque (k1) (Nm)",
              ylabel="degrees")


def plot_const_force_spring_vs_max_th1():
    count = 10
    spring_taus = range(0, count)
    th1s = np.zeros(count)

    for tau in spring_taus:
        robot = Robot()
        robot.c.motor_torque = 0
        robot.c.k1 = 0
        robot.c.spring_const_torque = tau
        data = simulate(robot,
                        th1_0=np.pi + 0.1,
                        dth1_0=0,
                        tmax=0.5,
                        debug_info=False)
        th1s[tau] = np.max(np.max(data.th_traj[0, :]))
        print(f"tau = {tau} th1 = {np.degrees(th1s[tau]):.2f} degrees")
        # animate_simulation(robot, data)
    th1s = th1s * 57.2958  #B convert to degrees

    plot_data(spring_taus,
              th1s,
              title="Const Force Spring Selection",
              xlabel="Spring Torque (Nm)",
              ylabel="degrees")


def run_animation():
    robot = Robot()
    data = simulate(robot, th1_0=np.pi + 0.1, dth1_0=0, tmax=1)
    animate_simulation(robot, data)
    # data.export_to_csv()


def get_spring_compare(count=20, k1=0, const_spring=0, extension=0):
    spring_taus = range(0, count)
    th1s = np.zeros(count)

    for tau in spring_taus:
        robot = Robot()
        robot.c.motor_torque = 0

        if (k1 != 0):
            robot.c.spring_gen_k = 0
            robot.c.spring_const_torque = 0
            robot.c.k1 = tau

        if (const_spring != 0):
            robot.c.spring_gen_k = 0
            robot.c.spring_const_torque = tau
            robot.c.k1 = 0

        if (extension != 0):
            robot.c.spring_gen_k = tau * 175.126835  #used imperial units for this since its sized against one
            robot.c.spring_const_torque = 0
            robot.c.k1 = 0

        data = simulate(robot,
                        th1_0=np.pi + 0.1,
                        dth1_0=0,
                        tmax=0.5,
                        debug_info=False)
        th1s[tau] = np.max(np.max(data.th_traj[0, :]))
        print(f"tau = {tau} th1 = {np.degrees(th1s[tau]):.2f} degrees")
        # animate_simulation(robot, data)
    th1s = th1s * 57.2958  #B convert to degrees

    return th1s, spring_taus


def get_heights_with_given_springs(count=100,
                                   k1=0,
                                   const_spring=0,
                                   extension=0):
    torques = range(0, count)
    jump_heights = np.zeros(count)

    for tau in torques:
        robot = Robot()
        robot.c.motor_torque = tau
        robot.c.spring_gen_k = extension * 57.2958
        robot.c.spring_const_torque = const_spring
        robot.c.k1 = k1

        data = simulate(robot,
                        th1_0=np.pi + 0.1,
                        dth1_0=0,
                        tmax=0.5,
                        debug_info=False)
        jump_heights[tau] = np.max(np.max(data.y_traj[0, :]) * 1000)
        print(f"tau = {tau} h = {jump_heights[tau]:.2f} mm")

    return jump_heights, torques


def plot_motor_vs_springs():
    jump_no_spring, t = get_heights_with_given_springs()
    jump_k1, t = get_heights_with_given_springs(k1=5)
    jump_const, t = get_heights_with_given_springs(const_spring=5)
    jump_extend, t = get_heights_with_given_springs(extension=5)

    plt.figure(figsize=(8, 5))
    plt.plot(t, jump_no_spring, label="No Spring")
    plt.plot(t, jump_k1, label="General Torsion Spring")
    plt.plot(t, jump_const, label="Constant Torsion Spring")
    plt.plot(t, jump_extend, label="Extension Spring")

    plt.xlabel("Motor Torque (Nm)")
    plt.ylabel("Jump Height (mm)")
    plt.title("Spring comparison with a constant k=5 value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_springs_find_k():
    jump_k1, t = get_spring_compare(k1=5)
    jump_const, t = get_spring_compare(const_spring=5)
    jump_extend, t = get_spring_compare(extension=5)

    plt.figure(figsize=(8, 5))
    plt.plot(t, jump_k1, label="General Torsion Spring")
    plt.plot(t, jump_const, label="Constant Torsion Spring")
    plt.plot(t, jump_extend, label="Extension Spring")

    plt.xlabel("k")
    plt.ylabel("q1 Angle (degrees)")
    plt.title("Solving for k of the spring")
    plt.legend()
    plt.grid(True)
    plt.show()


# main function to run the simulation and animation
run_animation()

# other ways of plotting the simulation data

# plot_springs_find_k()

# plot_motor_vs_springs()

# plot_motor_torque_vs_height()

# plot_spring_vs_max_th1()

# plot_const_force_spring_vs_max_th1()

# ---- plot some info about the data from the simulation ----

# robot = Robot()
# data = simulate(robot, th1_0=np.pi + 0.1, dth1_0=0, tmax=1)

# plot_data(data.time,
#           data.y_traj[1, :],
#           title="COM Y Velocity",
#           xlabel="Time (ms)",
#           ylabel="m/s")

# plot_data(data.time,
#           data.y_traj[0, :] * 1000,
#           title="Jump Height",
#           xlabel="Time (ms)",
#           ylabel="Hight (mm)")

# plot_data(data.time,
#           data.th_traj[0, :],
#           title="th1",
#           xlabel="Time (ms)",
#           ylabel="radians")

# plot_data(data.time,
#           data.th_traj[1, :],
#           title="dth1",
#           xlabel="Time (ms)",
#           ylabel="radians/s")
