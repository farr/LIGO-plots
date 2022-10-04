# Example of running `ptemcee`-based Bilby sampling.

To run GW150914 following the official Bilby example, you can issue 

```shell
python /path/to/LIGO-plots/main/scripts/bilby_ptemcee --ini config.ini
```

in the current directory.  Output will appear in `outdir`, including checkpoint
files (`checkpoint.nc`) and various status plots as the run progresses:

- `beta.png` shows the evolution of the inverse temperatures of the parallel
  chains versus iteration number as they adapt to the posterior surface.  At
  first the temperatures will move around alot as they seek to equilibrate the
  transitions between temperatures in the PT algorithm; eventually they should
  converge to good values and hold steady for the remainder of the run.
- `mean-likelihood.png` shows the mean of the likelihood (actually posterior,
  but who's keeping track?) for the `T = 1` (cold) chain's ensemble versus
  iteration number.  Early on, this will be increasing, as the ensemble moves to
  high-likelihood positions; later on, this will fluctuate randomly, as the
  ensemble explores the high-likelihood part of parameter space.
- `trace.png` shows a traceplot of the evolution of each walker in each
  parameter over each iteration.  Once the walkers start to explore the
  high-likelihood region of parameter space, and the total number of iterations
  (always 128*`thin` with `thin` increasing until the sampling converges)
  increases, this will get smoother and smoother.
- `flat-trace.png` shows a "flattened" trace where all walkers have been jammed
  together into a single chain; this ultimately reflects the posterior density.
- `ensemble-means.png` shows how the mean of the ensemble of walkers evolves
  over the sampling (in standardized---mean zero, standard deviation
  1---coordinates).  During burnin there will be trends in the evolution of the
  mean in some parameters; after burnin, but before convergence, there will be
  correlations from sample to sample, but not evolution; and once the sampler is
  burned in the mean will fluctuate randomly from sample to sample.

The run will go through a number of phases (sometimes reverting to earlier
phases if it detects the necessity):
- Initially there is a short interval between checkpoints as the walkers move to
  the high-likelihood places in parameter space.  Short runs will be repeated as
  long as the average likelihood keeps increasing.
- Once the high-likelihood region is found, walkers start exploring.  In this
  phase, we run for longer and longer stretches between checkpoints /
  convergence tests, doubling `thin` and the total number of MCMC steps at each
  iteartion (so the number of *saved* steps remains constant).  Eventually
  either a new high-likelihood region is found (in which case we revert to the
  initial phase again), or the autocorrelation length calculation shows that the
  chains have converged (in which case we stop the run).  We print the estimated
  autocorrelation length in each parameter at the end of each iteration in this
  phase; convergence occurs when the chains reach 50 autocorrelation lengths in
  each parameter.
- Finally we output the chains (sampled-over parameters plus various computed
  parameters) to `posterior.nc` in `arviz`-readable format (see
  [here](https://python.arviz.org/en/latest/)) for subsequent processing.

By default, the sampler will try to start first from the checkpointed position
of a previous run in `{outdir}/checkpoint.nc` (you can disable by setting
`try_checkpoint:False` in ini file); if you want a fresh run, you need to delete
this file or specify a new output directory.  Note: this is not a *true*
checkpoint---the run starts over---but just using the stored positions in
`checkpoint.nc` to initialize the sampler!

By default, the sampler will use the coordinate systems from my special branch
of Bilby, at https://git.ligo.org/will-farr/bilby/-/tree/spherical-coordinates;
you will need to install Bilby from that branch, or else disable the spherical
topology (leading to less-efficient `ptemcee` sampling, alas!) by setting
`spherical_topology:False` in the ini file).

Set the `pool_size` option in the ini file to the number of cores you want to
use.  Each step in the evolution of all the walkers involves `nwalkers*ntemps`
likelihood calls (the default settings have `64*8 = 512` likelihood calls per
step).  As long as there are several likelihood calls per core (i.e. `pool_size
<< nwalkers*ntemps`), there is minimal communication overhead, so the speedup
should be linear in the number of cores.

For the full list of `ini` file options, see the beginning of
`scripts/bilby_ptemcee.py`.