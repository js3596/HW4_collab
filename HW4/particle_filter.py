import numpy as np
import scipy.linalg  # You may find scipy.linalg.block_diag useful
import scipy.stats  # You may find scipy.stats.multivariate_normal.pdf useful
from . import turtlebot_model as tb

EPSILON_OMEGA = 1e-3

class ParticleFilter(object):
    """
    Base class for Monte Carlo localization and FastSLAM.

    Usage:
        pf = ParticleFilter(x0, R)
        while True:
            pf.transition_update(u, dt)
            pf.measurement_update(z, Q)
            localized_state = pf.x
    """

    def __init__(self, x0, R):
        """
        ParticleFilter constructor.

        Inputs:
            x0: np.array[M,3] - initial particle states.
             R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.M = x0.shape[0]  # Number of particles
        self.xs = x0  # Particle set [M x 3]
        self.ws = np.repeat(1. / self.M, self.M)  # Particle weights (initialize to uniform) [M]
        self.R = R  # Control noise covariance (corresponding to dt = 1 second) [2 x 2]

    @property
    def x(self):
        """
        Returns the particle with the maximum weight for visualization.

        Output:
            x: np.array[3,] - particle with the maximum weight.
        """
        idx = self.ws == self.ws.max()
        x = np.zeros(self.xs.shape[1:])
        x[:2] = self.xs[idx,:2].mean(axis=0)
        th = self.xs[idx,2]
        x[2] = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        return x

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating self.xs.

        Inputs:
            u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.xs) should be updated.
        """
        ########## Code starts here ##########
        # TODO: Update self.xs.
        # Hint: Call self.transition_model().
        # Hint: You may find np.random.multivariate_normal useful.

        new_u = np.random.multivariate_normal(u,self.R,self.M)
        self.xs = self.transition_model(new_u,dt)

        ########## Code ends here ##########

    def transition_model(self, us, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        raise NotImplementedError("transition_model must be overridden by a subclass of EKF")

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        raise NotImplementedError("measurement_update must be overridden by a subclass of EKF")

    def resample(self, xs, ws):
        """
        Resamples the particles according to the updated particle weights.

        Inputs:
            xs: np.array[M,3] - matrix of particle states.
            ws: np.array[M,]  - particle weights.

        Output:
            None - internal belief state (self.xs, self.ws) should be updated.
        """
        r = np.random.rand() / self.M

        ########## Code starts here ##########
        # TODO: Update self.xs, self.ws.
        # Note: Assign the weights in self.ws to the corresponding weights in ws
        #       when resampling xs instead of resetting them to a uniform
        #       distribution. This allows us to keep track of the most likely
        #       particle and use it to visualize the robot's pose with self.x.
        # Hint: To maximize speed, try to implement the resampling algorithm
        #       without for loops. You may find np.linspace(), np.cumsum(), and
        #       np.searchsorted() useful. This results in a ~10x speedup.


        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction).

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2I,]   - joint measurement mean.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        raise NotImplementedError("measurement_model must be overridden by a subclass of EKF")


class MonteCarloLocalization(ParticleFilter):

    def __init__(self, x0, R, map_lines, tf_base_to_camera, g):
        """
        MonteCarloLocalization constructor.

        Inputs:
                       x0: np.array[M,3] - initial particle states.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[2,J] - J map lines in columns representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as columns
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, R)

    def transition_model(self, us, dt):
        """
        Unicycle model dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """

        ########## Code starts here ##########
        # TODO: Compute g.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: A simple solution can be using a for loop for each partical
        #       and a call to tb.compute_dynamics
        # Hint: To maximize speed, try to compute the dynamics without looping
        #       over the particles. If you do this, you should implement
        #       vectorized versions of the dynamics computations directly here
        #       (instead of modifying turtlebot_model). This results in a
        #       ~10x speedup.
        # Hint: This faster/better solution does not use loop and does 
        #       not call tb.compute_dynamics. You need to compute the idxs
        #       where abs(om) > EPSILON_OMEGA and the other idxs, then do separate 
        #       updates for them

        # empty g
        g = np.zeros((self.M,3))

        # find idx of abs(om)<epsilon_omega and abs(om)<epsilon_omega
        om_ls = np.array(us[:,1])
        less_om_ls = abs(om_ls)<=EPSILON_OMEGA
        great_om_ls = abs(om_ls)>EPSILON_OMEGA
        less_idx = np.where(less_om_ls)[0]
        great_idx = np.where(great_om_ls)[0]

        # calculate g abs(om)<epsilon_omega
        us_less_v = us[less_idx,0]
        x_less_x = self.xs[less_idx,0]
        x_less_y = self.xs[less_idx,1]
        x_less_th = self.xs[less_idx,2]
        g[less_idx,0] = x_less_x+us_less_v*np.cos(x_less_th)*dt #x_t
        g[less_idx,1] = x_less_y+us_less_v*np.sin(x_less_th)*dt #y_t

        # calculate g abs(om)>epsilon_omega
        us_great_v = us[great_idx,0]
        us_great_om = us[great_idx,1]
        x_great_x = self.xs[great_idx,0]
        x_great_y = self.xs[great_idx,1]
        x_great_th = self.xs[great_idx,2]
        g[great_idx,0] = x_great_x+us_great_v/us_great_om*(np.sin(x_great_th+us_great_om*dt)-np.sin(x_great_th)) #x_t
        g[great_idx,1] = x_great_y-us_great_v/us_great_om*(np.cos(x_great_th+us_great_om*dt)-np.cos(x_great_th)) #y_t

        # th_t
        g[:,2] = self.xs[:,2]+us[:,1]*dt
        ########## Code ends here ##########

        return g

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        xs = np.copy(self.xs)
        ws = np.zeros_like(self.ws)

        ########## Code starts here ##########
        # TODO: Compute new particles (xs, ws) with updated measurement weights.
        # Hint: To maximize speed, implement this without looping over the
        #       particles. You may find scipy.stats.multivariate_normal.pdf()
        #       useful.
        # Hint: You'll need to call self.measurement_model()


        ########## Code ends here ##########

        self.resample(xs, ws)

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature for each particle.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[M,2I]  - joint measurement mean for M particles.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        vs = self.compute_innovations(z_raw, np.array(Q_raw))

        ########## Code starts here ##########
        # TODO: Compute Q.
        # Hint: You might find scipy.linalg.block_diag() useful


        ########## Code ends here ##########

        return vs, Q

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: np.array[I,2,2] - I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            vs: np.array[M,2I] - M innovation vectors of size 2I
                                 (predicted map measurement - scanner measurement).
        """
        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        ########## Code starts here ##########
        # TODO: Compute vs (with shape [M x I x 2]).
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       observed line, find the most likely map entry (the entry with 
        #       least Mahalanobis distance).
        # Hint: To maximize speed, try to eliminate all for loops, or at least
        #       for loops over J. It is possible to solve multiple systems with
        #       np.linalg.solve() and swap arbitrary axes with np.transpose().
        #       Eliminating loops over J results in a ~10x speedup.
        #       Eliminating loops over I results in a ~2x speedup.
        #       Eliminating loops over M results in a ~5x speedup.
        #       Overall, that's 100x!
        # Hint: For the faster solution, you might find np.expand_dims(), 
        #       np.linalg.solve(), np.meshgrid() useful.

        # compute hs matrix [M,2,J]
        hs = self.compute_predicted_measurements()

        # dimension constants
        m = self.M
        i = len(z_raw[0])
        j = len(hs[0][0])

        # calculate vij_a and vij_r [M,I,J]
        hs_a = hs[:,0,:] #[M,J]
        hs_a_calc = np.reshape(np.repeat(hs_a,i,axis=0),(m,i,j)) #[M,I,J]
        hs_r = hs[:,1,:] #[M,J]
        hs_r_calc = np.reshape(np.repeat(hs_r,i,axis=0),(m,i,j)) #[M,I,J]
        z_a = z_raw[0] #[1,I]
        z_a_calc = np.reshape(np.repeat(np.tile(z_a,(m,1)),j),(m,i,j)) #[M,I,J]
        z_r = z_raw[1] #[1,I]
        z_r_calc = np.reshape(np.repeat(np.tile(z_r,(m,1)),j),(m,i,j)) #[M,I,J]
        # vij_r [M,I,J]
        vij_r = z_r_calc-hs_r_calc
        # vij_a [M,I,J]
        vij_a = angle_diff(z_a_calc,hs_a_calc)

        # get Q_new [2,2,M,I,J]
        Q_i_0_0 = Q_raw[:,0,0]
        Q_i_0_1 = Q_raw[:,0,1]
        Q_i_1_0 = Q_raw[:,1,0]
        Q_i_1_1 = Q_raw[:,1,1]
        Q_new = np.zeros((2,2,m,i,j))
        Q_new[0,0] = np.reshape(np.repeat(np.tile(Q_i_0_0,(m,1)),j),(m,i,j))
        Q_new[0,1] = np.reshape(np.repeat(np.tile(Q_i_0_1,(m,1)),j),(m,i,j))
        Q_new[1,0] = np.reshape(np.repeat(np.tile(Q_i_1_0,(m,1)),j),(m,i,j))
        Q_new[1,1] = np.reshape(np.repeat(np.tile(Q_i_1_1,(m,1)),j),(m,i,j))


        # calculate dij [M,I,J]
        v_tot = np.array([vij_a,vij_r])
        dij = v_tot.T @ np.linalg.inv(Q_new) @ v_tot
        min_d = np.where(dij[:,:,]==min(dij[:,:,]))
        idx_M = min_d[0]
        idx_I = min_d[1]
        idx_J = min_d[2]

        # calculate vs [M,I,2]
        vs = np.zeros((m,i,2))
        vs[:,:,0] = vij_a[idx_M,idx_I,idx_J]
        vs[:,:,1] = vij_r[idx_M,idx_I,idx_J]
        ########## Code ends here ##########

        # Reshape [M x I x 2] array to [M x 2I]
        return vs.reshape((self.M,-1))  # [M x 2I]

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Output:
            hs: np.array[M,2,J] - J line parameters in the scanner (camera) frame for M particles.
        """
        ########## Code starts here ##########
        # TODO: Compute hs.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       map line, transform to scanner frmae using tb.transform_line_to_scanner_frame()
        #       and tb.normalize_line_parameters()
        # Hint: To maximize speed, try to compute the predicted measurements
        #       without looping over the map lines. You can implement vectorized
        #       versions of turtlebot_model functions directly here. This
        #       results in a ~10x speedup.
        # Hint: For the faster solution, it does not call tb.transform_line_to_scanner_frame()
        #       or tb.normalize_line_parameters(), but reimplement these steps vectorized.
        
        # empty hs [M,2,J]
        j = len(self.map_lines[0])
        hs = np.zeros((self.M,2,j))

        # all a and r [1, J]
        a0 = self.map_lines[0]
        r0 = self.map_lines[1]
        
        # make a and r into [M,J]
        a = np.tile(a0,(self.M,1))
        r = np.tile(r0,(self.M,1))

        # coordinate transform from world to body [1,M]
        x_wb = self.xs[:,0]
        y_wb = self.xs[:,1]
        th_wb = self.xs[:,2]

        # coordinate transform from body to camera [1,]
        x_bc = self.tf_base_to_camera[0]
        y_bc = self.tf_base_to_camera[1]
        th_bc = self.tf_base_to_camera[2]

        # R [M,2,2]
        R = np.zeros((self.M,2,2))
        R[:,0,0] = np.cos(th_wb)
        R[:,0,1] = np.sin(th_wb)
        R[:,1,0] = -np.sin(th_wb)
        R[:,1,1] = np.cos(th_wb)

        # compute hs
        hs[:,0,:] = a-np.tile(th_wb,(j,1)).T-th_bc # a
        term_cos = np.tile(x_wb+x_bc*R[:,0,0]+y_bc*R[:,1,0],(j,1)).T
        term_sin = np.tile(y_wb+x_bc*R[:,0,1]+y_bc*R[:,1,1],(j,1)).T
        hs[:,1,:] = r-term_cos*np.cos(a)-term_sin*np.sin(a) # r

        # normalization
        r_less_idx = np.where(hs[:,1,:]<0)
        idx_M = r_less_idx[0]
        idx_J = r_less_idx[1]
        hs[idx_M,1,idx_J] *= -1
        hs[idx_M,0,idx_J] += np.pi
        hs[:,0,:] = (hs[:,0,:]+np.pi)%(2*np.pi)-np.pi
        
                  
        
        ########## Code ends here ##########

        return hs

