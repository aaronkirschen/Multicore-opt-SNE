
from glob import glob
import threading
import os
import sys

import numpy as np
import cffi

'''
    Helper class to execute TSNE in separate thread.
'''


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)


class MulticoreTSNE:
    """
    Compute t-SNE embedding using Barnes-Hut optimization and
    multiple cores (if available).

    Parameters mostly correspond to parameters of `sklearn.manifold.TSNE`.

    The following parameters are unused:
    * n_iter_without_progress
    * min_grad_norm
    * metric
    * method

    When `cheat_metric` is true squared Euclidean distance is used to build VPTree.
    Usually leads to the same quality, yet much faster.

    Parameter `init` doesn't support 'pca' initialization, but a precomputed
    array can be passed.

    Parameter `n_iter_early_exag` defines the number of iterations out of total `n_iter`
    to spend in the early exaggeration phase of the algorithm. With the default `learning_rate`,
    the default values of 250/1000 may need to be increased when embedding large numbers
    of observations. Properly setting `learning_rate` results in good embeddings with fewer
    iterations. This interplay is discussed at https://doi.org/10.1101/451690.

    Parameter `auto_iter`, when set to true, causes the algorithm to ignore the `n_iter`
    and `n_iter_early_exag` parameters, and instead determine them dynamically. See readme
    for details.

    Parameter `auto_iter_end` is the constant used to stop the run when (KLDn-1 – KLDn) < KLDn/X
    where X is this arg. Only used when `auto_iter` is true.

    Parameter `optimize_perplexity`, when set to true, enables the optimization of the perplexity
    parameter within the range defined by `min_perplexity`, `max_perplexity`, and `step`.

    Parameters
    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.

    perplexity : float, optional (default: 30.0)
        The perplexity is related to the number of nearest neighbors that is used in other manifold
        learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly different results.

    early_exaggeration : float, optional (default: 12.0)
        Controls how tight natural clusters in the original space are in the embedded space and how much space
        will be between them. For larger values, the space between natural clusters will be larger in the
        embedded space. Again, the choice of this parameter is not very critical. If the cost function increases
        during initial optimization, the early exaggeration factor or the learning rate might be too high.

    learning_rate : float, optional (default: 200.0)
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the
        data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense cloud with few outliers. If the cost
        function increases during initial optimization, the learning rate might be too high. In that case, the
        learning rate should be decreased.

    n_iter : int, optional (default: 1000)
        Maximum number of iterations for the optimization. Should be at least 250.

    n_iter_early_exag : int, optional (default: 250)
        Number of iterations to spend in the early exaggeration phase.

    n_iter_without_progress : int, optional (default: 30)
        This parameter is not used in this implementation.

    min_grad_norm : float, optional (default: 1e-7)
        This parameter is not used in this implementation.

    metric : string or callable, optional (default: 'euclidean')
        This parameter is not used in this implementation.

    init : string or numpy array, optional (default: 'random')
        Initialization of embedding. Possible options are 'random', 'pca', and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with this implementation.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or None, optional (default: None)
        If int, random_state is the seed used by the random number generator; If None, the random number generator is the
        RandomState instance used by np.random.

    method : string, optional (default: 'barnes_hut')
        This parameter is not used in this implementation.

    angle : float, optional (default: 0.5)
        Trade-off between speed and accuracy for Barnes-Hut T-SNE. If set to 0.0, the Barnes-Hut algorithm is exact.
        If set to 1.0, it is very approximate.

    n_jobs : int, optional (default: 1)
        The number of parallel jobs to run for neighbors search. This parameter has no effect if it is not supported by the backend.

    cheat_metric : bool, optional (default: True)
        When true squared Euclidean distance is used to build VPTree. Usually leads to the same quality, yet much faster.

    auto_iter : bool, optional (default: False)
        When true, the algorithm will determine the number of iterations and early exaggeration iterations dynamically.

    auto_iter_end : int, optional (default: 5000)
        The constant used to stop the run when (KLDn-1 – KLDn) < KLDn/X where X is this parameter. Only used when `auto_iter` is true.

    optimize_perplexity : bool, optional (default: False)
        When true, enables the optimization of the perplexity parameter within the range defined by `min_perplexity`, `max_perplexity`, and `step`.

    min_perplexity : float, optional (default: 5.0)
        Minimum perplexity value for optimization. Used only if `optimize_perplexity` is true.

    max_perplexity : float, optional (default: 50.0)
        Maximum perplexity value for optimization. Used only if `optimize_perplexity` is true.

    step : float, optional (default: 5.0)
        Step size for perplexity optimization. Used only if `optimize_perplexity` is true.

    Attributes
    ----------
    embedding_ : array, shape (n_samples, n_components)
        Stores the embedding.

    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.

    n_iter_ : int
        Number of iterations run.

    optimized_perplexity : float or None
        Stores the optimized perplexity value if `optimize_perplexity` is True, otherwise None.

    """
    def __init__(self,
                 n_components=2,
                 perplexity=30.0,
                 early_exaggeration=12,
                 learning_rate=200,
                 n_iter=1000,
                 n_iter_early_exag=250,
                 n_iter_without_progress=30,
                 min_grad_norm=1e-07,
                 metric='euclidean',
                 init='random',
                 verbose=0,
                 random_state=None,
                 method='barnes_hut',
                 angle=0.5,
                 n_jobs=1,
                 cheat_metric=True,
                 auto_iter=False,
                 auto_iter_end=5000,
                 optimize_perplexity=False,
                 min_perplexity=5.0,
                 max_perplexity=50.0,
                 step=5.0):
        self.n_components = n_components
        self.angle = angle
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_early_exag = n_iter_early_exag
        self.n_jobs = n_jobs
        self.random_state = -1 if random_state is None else random_state
        self.init = init
        self.embedding_ = None
        self.n_iter_ = None
        self.kl_divergence_ = None
        self.verbose = int(verbose)
        self.cheat_metric = cheat_metric
        self.auto_iter = auto_iter
        self.auto_iter_end = auto_iter_end
        self.optimize_perplexity = optimize_perplexity
        self.min_perplexity = min_perplexity
        self.max_perplexity = max_perplexity
        self.step = step
        self.optimized_perplexity = None

        assert isinstance(init, np.ndarray) or init == 'random', "init must be 'random' or array"
        if isinstance(init, np.ndarray):
            assert init.ndim == 2, "init array must be 2D"
            assert init.shape[1] == n_components, "init array must be of shape (n_instances, n_components)"
            self.init = np.ascontiguousarray(init, float)

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            """void tsne_run_double(double* X, int N, int D, double* Y,
                                    int no_dims, double perplexity, double theta,
                                    int num_threads, int max_iter, int n_iter_early_exag,
                                    int random_state, bool init_from_Y, int verbose,
                                    double early_exaggeration, double learning_rate,
                                    double *final_error, int distance, bool auto_iter, double auto_iter_end,
                                    bool optimize_perplexity, double min_perplexity, double max_perplexity, double step,
                                    double *optimized_perplexity);"""
        )


        path = os.path.dirname(os.path.realpath(__file__))
        try:
            sofile = (glob(os.path.join(path, 'libtsne*.so')) +
                      glob(os.path.join(path, '*tsne*.dll')))[0]
            self.C = self.ffi.dlopen(os.path.join(path, sofile))
        except (IndexError, OSError):
            raise RuntimeError('Cannot find/open tsne_multicore shared library')

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, _y=None):

        assert X.ndim == 2, 'X should be 2D array.'
        # X may be modified, make a copy
        X = np.array(X, dtype=float, order='C', copy=True)

        N, D = X.shape
        init_from_Y = isinstance(self.init, np.ndarray)
        if init_from_Y:
            Y = self.init.copy('C')
            assert X.shape[0] == Y.shape[0], "n_instances in init array and X must match"
        else:
            Y = np.zeros((N, self.n_components))

        cffi_X = self.ffi.cast('double*', X.ctypes.data)
        cffi_Y = self.ffi.cast('double*', Y.ctypes.data)
        final_error = np.array(0, dtype=float)
        cffi_final_error = self.ffi.cast('double*', final_error.ctypes.data)
        optimized_perplexity = np.array(self.perplexity, dtype=float)
        cffi_optimized_perplexity = self.ffi.cast('double*', optimized_perplexity.ctypes.data)

        t = FuncThread(self.C.tsne_run_double,
                       cffi_X, N, D,
                       cffi_Y, self.n_components,
                       self.perplexity, self.angle, self.n_jobs, self.n_iter, self.n_iter_early_exag,
                       self.random_state, init_from_Y, self.verbose, self.early_exaggeration,
                       self.learning_rate, cffi_final_error, int(self.cheat_metric), self.auto_iter, self.auto_iter_end,
                       self.optimize_perplexity, self.min_perplexity, self.max_perplexity, self.step,
                       cffi_optimized_perplexity)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        self.embedding_ = Y
        self.kl_divergence_ = final_error
        self.n_iter_ = self.n_iter

        if self.optimize_perplexity:
            self.optimized_perplexity = optimized_perplexity

        return Y
