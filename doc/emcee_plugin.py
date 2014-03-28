from glue.core.fitters import BaseFitter1D
from glue.core.simpleforms import IntOption
from glue.config import fit_plugin

import numpy as np
import emcee


def gaussian(x, mean, amplitude, stddev):
    return np.exp(-(x - mean) ** 2 / (2 * stddev ** 2)) * amplitude


def lnprob(param, x, y, dy):
    # mean, amplitude, stddev = param
    if param[-1] < 0:
        return -np.inf
    yp = gaussian(x, *param)
    diff = (y - yp)
    if dy is not None:
        diff /= dy
    return -np.dot(diff, diff)


@fit_plugin
class EmceeGaussianFitter(BaseFitter1D):
    label = "Emcee Gaussian"
    walkers = IntOption(label="Walkers", min=1, max=200, default=50)
    burnin = IntOption(label="Burn in steps", min=1, max=10000, default=500)
    steps = IntOption(label="Steps", min=10, max=10000, default=500)

    def fit(self, x, y, dy, constraints,
            walkers=50, burnin=500, steps=500):
        ndim = 3
        # initialize walker parameters
        amp = y.max()
        mean = (x * y).sum() / y.sum()
        var = ((x - mean) ** 2 * y).sum() / y.sum()
        p0 = np.array([amp, mean, np.sqrt(var)]).reshape(1, -1)
        p0 = np.random.lognormal(sigma=.1, size=(walkers, ndim)) * p0
        sampler = emcee.EnsembleSampler(walkers, ndim, lnprob,
                                        args=[x, y, dy])

        # burnin
        pos, prob, state = sampler.run_mcmc(p0, burnin)
        sampler.reset()

        # run
        sampler.run_mcmc(pos, steps)
        return sampler

    def predict(self, fit_result, x):
        chain = fit_result.flatchain
        params = np.mean(chain, axis=0)
        return gaussian(x, *params)

    def summarize(self, fit_result, x, y, dy):
        af = fit_result.acceptance_fraction.mean()
        chain = fit_result.flatchain
        amp, mean, sigma = chain.mean(axis=0)
        damp, dmean, dsigma = np.std(chain, axis=0)
        walkers, steps, dim = fit_result.chain.shape

        result = [
            "Walkers: %i" % walkers,
            "Steps:   %i" % steps,
            "Acceptance fraction: %0.2f" % af,
            "-------------------------",
            "amplitude = %0.3e +/- %0.1e" % (amp, damp),
            "mean      = %0.3e +/- %0.1e" % (mean, dmean),
            "stddev    = %0.3e +/- %0.1e" % (sigma, dsigma)
        ]
        return '\n'.join(result)

    def plot(self, fit_result, axes, x):
        chain = fit_result.flatchain
        result = []

        # background samples
        for i in range(100):
            row = np.random.randint(0, chain.shape[0])
            params = chain[row]
            y = gaussian(x, *params)
            result.extend(axes.plot(x, y, 'k', alpha=.08))

        # foreground prediction of posterior mean model
        result.extend(
            super(EmceeGaussianFitter, self).plot(fit_result, axes, x))

        return result
