"""ODE solver as proposed by Conrad et al."""
import numpy as np

import probnum.diffeq as pnd
import probnum.diffeq.odefiltsmooth.kalman_odesolution as odesol
import probnum.randvars as pnrv
from probnum import filtsmooth, problems, statespace
from probnum.diffeq.perturbedsolvers import perturbedstatesolution
from probnum.statespace import discrete_transition


class NoisyStateSolver(pnd.ODESolver):
    """ODE Solver based on Scipy that introduces uncertainty by adding Gaussian-noise
    with error-estimation dependant variance."""

    # pylint: disable=maybe-no-member
    def __init__(self, solver, noise_scale):
        self.solver = solver
        self.noise_scale = noise_scale
        self.perturbation = None
        self.interpolants = None
        self.kalman_odesolutions = None
        self.gauss_filter = None
        self.stepsize = None
        super().__init__(ivp=solver.ivp, order=solver.order)

    def initialise(self):
        """initialise the solver."""
        self.perturbation = []
        self.interpolants = []
        self.kalman_odesolutions = []
        # spatialdimension of the ODE
        spatialdim = self.solver.ivp.dimension
        # define dynamics model, measurement model and kalman filter
        dynamics_model = statespace.IBM(
            ordint=self.solver.order,
            spatialdim=spatialdim,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
        measurement_model = discrete_transition.DiscreteLTIGaussian(
            dynamics_model.proj2coord([0]),
            np.zeros(spatialdim),
            np.zeros([spatialdim, spatialdim]),
            proc_noise_cov_cholesky=np.zeros([spatialdim, spatialdim]),
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
        initsize = len(dynamics_model.proj2coord([0])[0])
        initrv = pnrv.Normal(
            np.zeros(initsize), np.eye(initsize), cov_cholesky=np.eye(initsize)
        )
        self.gauss_filter = filtsmooth.Kalman(dynamics_model, measurement_model, initrv)
        return self.solver.initialise()

    def step(self, start, stop, current, **kwargs):
        """performs one step from timepoint start to timepoint stop
        Parameters
        ----------
        start : float
            start point of the step
        stop : float
            stop point of the step
        current : :obj:list of :obj:RandomVariable
            current state at timepoint start

        Returns
        -------
        random_var : :obj:list of :obj:RandomVariable
            current state with perturbation after one step was performed
        error_estimation : float
            scipy error estimation after the performance of one step

        """
        proposed_rv, error_estimation = self.solver.step(start, stop, current)
        zero_mean = np.zeros(len(proposed_rv.mean))
        norm_distr_sample = pnrv.Normal(
            zero_mean, np.eye(len(proposed_rv.mean))
        ).sample(size=1)
        # for vector valued error estimation @ instead of *
        noise = error_estimation * self.noise_scale * norm_distr_sample
        random_var = pnrv.Constant(proposed_rv.mean + noise)
        self.perturbation.append(noise)
        self.stepsize = stop - start
        return random_var, error_estimation

    def dense_output(self):
        """calculate scipy_solver dense_output and and a kalman posterior that
        represents the perturbation.

        Returns
        -------
        unperturbed_dense_output : :obj:'scipy.integrate._ivp.base.DenseOutput`
            local interpolant over step made by an ODE solver
        kalman_ode_solution : :obj:'probnum.diffeq.odefiltsmooth.kalman_odesolution'
            Gauss-Markov posterior over the ODE solver state space model which
            represents the perturbation
        """
        unperturbed_dense_output = self.solver.dense_output()
        # calculate kalman posterior of the perturbations
        noise = self.perturbation[-1]
        time = self.stepsize
        states = np.asarray(
            [np.zeros(self.solver.ivp.dimension) + 0.00001, np.asarray(noise)]
        )
        times = np.asarray([0, time])
        regression_problem = problems.RegressionProblem(states, times)
        kalman_posterior = self.gauss_filter.filtsmooth(regression_problem)
        kalman_ode_solution = odesol.KalmanODESolution(kalman_posterior)
        return unperturbed_dense_output, kalman_ode_solution

    def method_callback(self, time, current_guess, current_error):
        """calculates dense output after each step and stores it in interpolants, stores
        the kalman posteriors in posteriors."""
        dense = self.dense_output()
        self.interpolants.append(dense[0])
        self.kalman_odesolutions.append(dense[1])

    def rvlist_to_odesol(self, times, rvs):
        interpolants = self.interpolants
        kalman_odesolutions = self.kalman_odesolutions
        probnum_solution = perturbedstatesolution.PerturbedStateSolution(
            times, rvs, interpolants, kalman_odesolutions, self.gauss_filter
        )
        return probnum_solution

    def postprocess(self, odesol):
        return self.solver.postprocess(odesol)
