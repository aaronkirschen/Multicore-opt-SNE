import unittest
from functools import partial

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from MulticoreTSNE import MulticoreTSNE


make_blobs = partial(make_blobs, random_state=0)
MulticoreTSNE = partial(MulticoreTSNE, random_state=3)


def pdist(X):
    """Condensed pairwise distances, like scipy.spatial.distance.pdist()"""
    return pairwise_distances(X)[np.triu_indices(X.shape[0], 1)]


class TestMulticoreTSNE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_blobs(n_samples=20, n_features=100, centers=2, shuffle=False)

    def test_tsne(self):
        X, y = self.X, self.y
        tsne = MulticoreTSNE(perplexity=5, n_iter=500)
        E = tsne.fit_transform(X)

        self.assertEqual(E.shape, (X.shape[0], 2))

        max_intracluster = max(pdist(E[y == 0]).max(),
                               pdist(E[y == 1]).max())
        min_intercluster = pairwise_distances(E[y == 0],
                                              E[y == 1]).min()

        self.assertGreater(min_intercluster, max_intracluster)

    def test_n_jobs(self):
        X, y = self.X, self.y
        tsne = MulticoreTSNE(n_iter=100, n_jobs=-2)
        tsne.fit_transform(X)

    def test_perplexity(self):
        X, y = self.X, self.y
        tsne = MulticoreTSNE(perplexity=X.shape[0], n_iter=100)
        tsne.fit_transform(X)

    def test_dont_change_x(self):
        X = np.random.random((20, 4))
        X_orig = X.copy()
        MulticoreTSNE(n_iter=400).fit_transform(X)
        np.testing.assert_array_equal(X, X_orig)

    def test_init_from_y(self):
        X, y = self.X, self.y
        tsne = MulticoreTSNE(n_iter=500)
        E = tsne.fit_transform(X)

        tsne = MulticoreTSNE(n_iter=0, init=E)
        E2 = tsne.fit_transform(X)
        np.testing.assert_allclose(E, E2)

        tsne = MulticoreTSNE(n_iter=1, init=E)
        E2 = tsne.fit_transform(X)
        mean_diff = np.abs((E - E2).sum(1)).mean()
        self.assertLess(mean_diff, 30)

    def test_attributes(self):
        X, y = self.X, self.y
        N_ITER = 200
        tsne = MulticoreTSNE(n_iter=N_ITER)
        E = tsne.fit_transform(X, y)

        self.assertIs(tsne.embedding_, E)
        self.assertGreater(tsne.kl_divergence_, 0)
        self.assertEqual(tsne.n_iter_, N_ITER)

    def test_optimize_perplexity(self):
        X, y = self.X, self.y
        tsne = MulticoreTSNE(perplexity=5, n_iter=500, optimize_perplexity=True, min_perplexity=5, max_perplexity=50, step=5)
        E = tsne.fit_transform(X)

        self.assertEqual(E.shape, (X.shape[0], 2))

        max_intracluster = max(pdist(E[y == 0]).max(),
                               pdist(E[y == 1]).max())
        min_intercluster = pairwise_distances(E[y == 0],
                                              E[y == 1]).min()

        self.assertGreater(min_intercluster, max_intracluster)

    def test_optimize_perplexity_values(self):
        X, y = self.X, self.y
        min_perplexity = 2
        max_perplexity = 5
        tsne = MulticoreTSNE(perplexity=1, optimize_perplexity=True, min_perplexity=min_perplexity, max_perplexity=max_perplexity, step=1, verbose=1)
        E = tsne.fit_transform(X)

        self.assertEqual(E.shape, (X.shape[0], 2))

        max_intracluster = max(pdist(E[y == 0]).max(),
                               pdist(E[y == 1]).max())
        min_intercluster = pairwise_distances(E[y == 0],
                                              E[y == 1]).min()

        self.assertGreater(min_intercluster, max_intracluster)

        # Check if perplexity is in the optimized range
        optimized_perplexity = tsne.optimized_perplexity
        print(f"Optimized Perplexity: {optimized_perplexity}")
        self.assertTrue(min_perplexity <= optimized_perplexity <= max_perplexity)

#%%
