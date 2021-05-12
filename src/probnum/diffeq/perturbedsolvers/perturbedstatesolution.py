import numpy as np

import probnum._randomvariablelist as pnrv_list
import probnum.diffeq as pnd
import probnum.randvars as pnrv
from probnum import problems
from probnum.diffeq.odefiltsmooth import kalman_odesolution


class PerturbedStateSolution(pnd.ODESolution):

    """Output of NoisyStateSolver.

    Parameters
    ----------
    times
        array of length N+1 where N is the number of steps that the ODE solver has taken.
    interpolants
        array of scipy.DenseOutput objects, shape (N,). Interpolates
        the deterministic
        solver output.
    kalman_odesolutions
        array of probnum.diffeq.KalmanODESolution objects, shape (N,).
        Interpolates the perturbations.
    """

    def __init__(self, times, states, interpolants, kalman_odesolutions, gauss_filter):
        self.interpolants = interpolants
        self.kalman_odesolutions = kalman_odesolutions
        self.kalman = gauss_filter
        super().__init__(locations=times, states=states)

    def __call__(self, t):
        if not np.isscalar(t):
            # recursive evaluation (t can now be any array, not just length 1!)
            return pnrv_list._RandomVariableList(
                [self.__call__(t_pt) for t_pt in np.asarray(t)]
            )
        # find closest timepoint (=í.e. correct interpolant) of evaluation
        closest_left_t = self.find_closest_left_element(self.times, t)
        # timepoint within the given interpolant
        interpolant_t = t - self.times[closest_left_t]
        interpolant = self.interpolants[closest_left_t]
        # evalution at timepoint t, not the interpolants' timepoint
        scipy_dense = interpolant(t)
        # sample from the Kalman-posterior
        kalpost = self.kalman_odesolutions[closest_left_t]
        noise_dense = kalpost.sample([interpolant_t])
        # insert t, noise_dense into ts and ys of the Kalman-posterior which
        # models the noise at correct position
        ys_perturbation = kalpost.states.mean
        ts_perturbation = kalpost.locations
        # check whether new y has to be inserted, insert on the left or right side
        ys_perturbation_new, ts_perturbation_new = self.insert_element(
            ts_perturbation, ys_perturbation, interpolant_t, noise_dense
        )
        regression_problem = problems.RegressionProblem(
            ys_perturbation_new, ts_perturbation_new
        )
        filtpost = self.kalman.filtsmooth(regression_problem)
        self.kalman_odesolutions[closest_left_t] = kalman_odesolution.KalmanODESolution(
            filtpost
        )
        output = scipy_dense + noise_dense
        return pnrv.Constant(output[0])

    @property
    def t(self):
        """Time points of the discrete-time solution."""
        return self.times

    @property
    def y(self):
        """Discrete-time solution."""
        return pnrv_list._RandomVariableList(self.states)

    def __len__(self) -> int:
        """Number of points in the discrete-time solution."""
        return len(self.states)

    def __getitem__(self, idx: int) -> pnrv.RandomVariable:
        """Access the :math:`i`th element of the discrete-time solution."""
        return self.states[idx]

    def find_closest_left_element(self, times, t_new):
        """

        Parameters
        ----------
        times : array
            array of discrete evaluation points [t_1, t_2, ...t_n]
        t_new : float
            timepoint t_new of which we want to find the closest left point in
            times

        Returns
        -------
        closest left timepoint of t in times, , i.e. if t_i < t_new < t_{i+1}
        the output it t_i. For t_n return t_{n-1}.

        """
        # find closest timepoint of evaluation
        closest_t = (np.abs(t_new - np.array(times))).argmin()
        # if t_new is in the first interpolant
        if t_new <= times[1]:
            closest_left_t = 0
        # make sure that the point is on the left of the evaluation point
        elif t_new <= times[closest_t]:
            closest_left_t = closest_t - 1
        else:
            closest_left_t = closest_t
        return closest_left_t

    def insert_element(self, times, states, t_new, state_new):
        """
        Parameters
        ----------
        times : array
            array of discrete evaluation points [t_1, t_2, ...t_n]
        states : array
            array of the corresponding states
        t_new : float
            timepoint t_new at which we want to insert state_new
        state_new : float
            new state y_new that should be inserted in states at position i+1
            if t_i < t_new < t_{i+1}


        Returns
        -------
        states and times with updated timestep and corresponding state respectively

        """
        closest_position = (np.abs(t_new - np.array(times))).argmin()
        if times[closest_position] != t_new:
            if t_new > times[closest_position]:
                insertion_pos = closest_position + 1
            else:
                insertion_pos = closest_position
            states = np.insert(states, insertion_pos, state_new, 0)
            times = np.insert(times, insertion_pos, t_new)
        return states, times
