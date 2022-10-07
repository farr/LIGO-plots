import argparse
from re import T
import arviz as az
import bilby
import configparser
import emcee
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os.path as op
import pandas as pd
import scipy.optimize as so
import sys
from tqdm import tqdm
import xarray as xr

def act_estimate(chain, c=5):
    nw, ns, nd = chain.shape

    taus = []
    for k in range(nd):
        mu = np.mean(chain[:,:,k])
        f = np.zeros(ns)
        for i in range(nw):
            n = 1
            while n < 2*nw:
                n = n << 1
            x = np.zeros(n)
            x[:ns] = chain[i,:,k] - mu
            f += np.fft.irfft(np.square(np.abs(np.fft.rfft(x))))[:ns]
        f = f / f[0]

        # From Vehtari, et al (2021)
        P = f[::2] + f[1::2]
        if np.any(P < 0):
            kk = np.argmin(P < 0)
        else:
            kk = len(P)
        taus.append(-1 + 2*np.sum(P[:kk]))

    return np.array(taus)


# Needed to make the multiprocessing work on my MacBook
if sys.platform.startswith('darwin'):
    multiprocessing.set_start_method('fork')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='program to run bilby with ptemcee')
    parser.add_argument('--ini', default='config.ini', help='default: %(default)s')
    
    args = parser.parse_args()

    cp = configparser.ConfigParser()
    cp.read(args.ini)
    cp = cp['DEFAULT']

    logger = bilby.core.utils.logger
    outdir = cp.get('outdir', 'outdir')
    label = cp.get('label', '')

    # Set to the number of cores you want to run on.
    pool_size = cp.getint('pool_size', 8)

    nw = cp.getint('nwalkers', 64)
    nt = cp.getint('ntemps', 8)
    niter = cp.getint('niter', 128)

    # If `conservative_convergence` is set, we consider converged when emcee's
    # integrated autocorrelation time is smaller than 1/50 of the chain (i.e.
    # chain is > 50 ACTs long); when `conservative_convergence` is `False`, we
    # consider converged when `arviz.ess` returns at least `converged_ess`
    # effective samples (default is 50% of the total samples).
    conservative_convergence = cp.getboolean('conservative_convergence', False)
    converged_ess = cp.getfloat('converged_ess', 0.9*nw*niter)

    try_checkpoint = cp.getboolean('try_checkpoint', True)

    trigger_time = cp.getfloat('trigger_time')
    
    detector_string = cp.get('detectors', 'H1,L1')
    detectors = detector_string.split(',')

    maximum_frequency = cp.getfloat('maximum_frequency', 512)
    minimum_frequency = cp.getfloat('minimum_frequency', 20)

    # Duration of the Tukey window in s.
    roll_off = cp.getfloat('rolloff', 0.4)

    # Duration of the analysis segment, s.
    duration = cp.getfloat('duration', 4)

    # Part of analysis segment after the trigger.
    post_trigger_duration = cp.getfloat('post_trigger_duration', 2)

    end_time = trigger_time + post_trigger_duration
    start_time = end_time - duration

    npsd = cp.getint('num_psd_segments', 32)
    psd_duration = npsd*duration

    psd_start_time = cp.getfloat('psd_start_time', start_time-psd_duration)
    psd_end_time = psd_start_time + psd_duration

    prior_file = cp.get('prior_file', 'GW150914.prior')

    # Use the standard Bilby coodinate system, which will make sampling very
    # inefficient.
    topology = cp.getboolean('spherical_topology', True)

    approximant = cp.get('approximant', 'IMRPhenomXPHM')
    fref = cp.getfloat('fref', 50)

    time_marginalization = cp.getboolean('time_marginalization', True)
    phase_marginalization = cp.getboolean('phase_marginalization', False)
    distance_marginalization = cp.getboolean('distance_marginalization', True)

    ifo_list = bilby.gw.detector.InterferometerList([])
    for det in detectors:
        logger.info("Downloading analysis data for ifo {}".format(det))
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        data = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        ifo.strain_data.set_from_gwpy_timeseries(data)

        logger.info("Downloading psd data for ifo {}".format(det))
        psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, cache=True)
        psd_alpha = 2 * roll_off / duration
        psd = psd_data.psd(
            fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
        )
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd.frequencies.value, psd_array=psd.value
        )
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
        ifo_list.append(ifo)

    logger.info("Saving data plots to {}".format(outdir))
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
    ifo_list.plot_data(outdir=outdir, label=label)

    priors = bilby.gw.prior.BBHPriorDict(filename=prior_file)

    # Add in the new coordinates---for this part you will need my branch from
    # https://git.ligo.org/will-farr/bilby/-/tree/spherical-coordinates You
    # *probably* don't really need it, but it will help with the acceptance rate
    # and convergence.  You can turn this off by setting `spherical_topology` to
    # `False` in the ini
    if topology:
        del priors['ra']
        del priors['dec']
        priors['sky_x'] = bilby.core.prior.Normal(0,1,name='sky_x')
        priors['sky_y'] = bilby.core.prior.Normal(0,1,name='sky_y')
        priors['sky_z'] = bilby.core.prior.Normal(0,1,name='sky_z')

        del priors['phase']
        priors['phase_x'] = bilby.core.prior.Normal(0,1,name='phase_x')
        priors['phase_y'] = bilby.core.prior.Normal(0,1,name='phase_y')

        del priors['theta_jn']
        del priors['psi']
        priors['rad_x'] = bilby.core.prior.Normal(0,1,name='rad_x')
        priors['rad_y'] = bilby.core.prior.Normal(0,1,name='rad_y')
        priors['rad_z'] = bilby.core.prior.Normal(0,1,name='rad_z')

        del priors['phi_12']
        priors['phi_12_x'] = bilby.core.prior.Normal(0,1,name='phi_12_x')
        priors['phi_12_y'] = bilby.core.prior.Normal(0,1,name='phi_12_y')

        del priors['phi_jl']
        priors['phi_jl_x'] = bilby.core.prior.Normal(0,1,name='phi_jl_x')
        priors['phi_jl_y'] = bilby.core.prior.Normal(0,1,name='phi_jl_y')

    priors["geocent_time"] = bilby.core.prior.Uniform(
        trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": approximant,
            "reference_frequency": fref,
    })

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator,
        priors=priors,
        time_marginalization=time_marginalization,
        phase_marginalization=phase_marginalization,
        distance_marginalization=distance_marginalization)

    # Don't ask me why, but this fails the first time through when
    # marginalization is turned on, but the second time through the marginalized
    # parameters will be set up 
    try:
        sampler = bilby.sampler.Ptemcee(likelihood, priors, outdir=outdir, label=label, 
                                        nwalkers=nw, ntemps=nt, 
                                        adapt=True, adaptation_time=10, adaptation_lag=100, Tmax=np.inf,
                                        plot=True, Q_tol=1.001, store_walkers=True)
    except KeyError:
        sampler = bilby.sampler.Ptemcee(likelihood, priors, outdir=outdir, label=label, 
                                        nwalkers=nw, ntemps=nt, 
                                        adapt=True, adaptation_time=10, adaptation_lag=100, Tmax=np.inf,
                                        plot=True, Q_tol=1.001, store_walkers=True)

    ptsampler = sampler.setup_sampler()
    sampler_kws = sampler.sampler_function_kwargs

    cfile = op.join(outdir, 'checkpoint.nc')
    nd = len(sampler.search_parameter_keys)
    if try_checkpoint and op.exists(cfile):
        pdata = az.from_netcdf(cfile)
        p0 = np.zeros((nt, nw, nd))
        d = pdata.posterior.draw[-1]
        c = pdata.posterior.chain
        for i, k in enumerate(sampler.search_parameter_keys):
            x = pdata.posterior[k].loc[dict(chain=c, draw=d)]
            p0[:,:,i] = np.tile(x, (nt, 1))
        pos0 = p0
        print('setup from checkpoint')
    else:
        pos0 = sampler.pos0

    # Setup for the sampling loop
    pos = pos0
    nt, nw, nd = pos.shape
    thin = 1
    max_mean_log_like = np.NINF
    converged = False

    def logprior(x):
        params = {k: t for k,t in zip(sampler.search_parameter_keys, x)}
        return sampler.priors.ln_prob(params)
    def loglike(x):
        params = {k: t for k,t in zip(sampler.search_parameter_keys, x)}
        likelihood.parameters.update(params)
        return likelihood.log_likelihood() - likelihood.noise_log_likelihood()

    global likelihood_fn
    likelihood_fn = loglike
    global prior_fn
    prior_fn = logprior

    class LikePriorEvaluator(object):
        def __init__(self):
            pass
        def __call__(self, x):
            lp = prior_fn(x)
            if lp == np.NINF:
                return np.NINF, np.NINF
            else:
                ll = likelihood_fn(x)
                return ll, lp

    with multiprocessing.Pool(processes=pool_size) as p:
        ptsampler.pool = p
        ptsampler._likeprior = LikePriorEvaluator()
        while not converged:
            ptsampler.reset()
            for pos, log_post, log_like in tqdm(ptsampler.sample(pos, storechain=True, iterations=niter*thin, thin=thin, **sampler_kws), total=niter*thin):
                pass

            print('Mean acceptance fraction:')
            print(np.mean(ptsampler.acceptance_fraction, axis=1))
            print()

            print('Mean tswap fraction:')
            print(ptsampler.tswap_acceptance_fraction)

            with open(op.join(outdir, 'stats.txt'), 'w') as f:
                f.write('Mean acceptance fraction:\n{}\n'.format(np.mean(ptsampler.acceptance_fraction, axis=1)))
                f.write('Mean tswap fraction:\n{}\n'.format(ptsampler.tswap_acceptance_fraction))

            nt, nw, ns, nd = ptsampler.chain.shape

            plt.figure()
            plt.plot(np.arange(ns), np.mean(ptsampler.logprobability[0,:,:], axis=0))
            plt.xlabel(r'Iteration')
            plt.ylabel(r'$\left\langle \log \pi \right\rangle_{\beta = 1}$')
            plt.savefig(op.join(outdir, 'mean-likelihood.png'))
            plt.close()

            plt.figure()
            plt.plot(ptsampler.beta_history.T)
            plt.xlabel(r'Iteration')
            plt.ylabel(r'$\beta$')
            plt.savefig(op.join(outdir, 'beta.png'))
            plt.close()

            idata = az.convert_to_inference_data({k: ptsampler.chain[0, :, :, i] for (i,k) in enumerate(sampler.search_parameter_keys)})
            az.to_netcdf(idata, op.join(outdir, 'checkpoint.nc'))

            az.plot_trace(idata)
            plt.savefig(op.join(outdir, 'trace.png'))
            plt.close()

            nt, nw, ns, nd = ptsampler.chain.shape
            flat_chain = ptsampler.chain.reshape((nt, 1, nw*ns, nd))
            flat_idata = az.convert_to_inference_data({k:flat_chain[0,:,:,i] for (i,k) in enumerate(sampler.search_parameter_keys)})
            az.plot_trace(flat_idata)
            plt.savefig(op.join(outdir, 'flat-trace.png'))
            plt.close()

            # Let's make a plot that shows the convergence (or lack thereof)
            plt.figure()
            c = ptsampler.chain[0,...]
            mu = np.mean(c, axis=0)
            mu_mu = np.mean(mu, axis=0)
            mu_std = np.std(mu, axis=0)
            for k in range(mu.shape[1]):
                plt.plot((mu[:,k]-mu_mu[k])/mu_std[k])
            plt.savefig(op.join(outdir, 'ensemble-means.png'))
            plt.close()

            lps = ptsampler.logprobability[0,:,:]
            mean_lps = np.mean(lps, axis=1) # Ensemble average at each timestep
            mean_log_like = np.mean(mean_lps) # Mean over timesteps
            se_log_like = np.std(mean_lps)/np.sqrt(len(mean_lps)) # Standard error of mean over timesteps.
            if mean_log_like > max_mean_log_like + 3*se_log_like:
                print('resetting sampler due to significant log-likelihood increase')
                print(f'current mean log(post) = {mean_log_like:.3f} (+/- {se_log_like:.3f})')
                print('is significantly larger than')
                print(f'previous best mean log(post) = {max_mean_log_like:.3f}')

                # Reset the sampler
                thin = 1
                max_mean_log_like = mean_log_like

                # Now sort the points by log-likelihood, and fill the chain in with only the highest (unique) points:
                all_pts = ptsampler.chain.reshape(-1, nd)
                all_logls = ptsampler.logprobability.flatten()

                _, u = np.unique(all_logls, return_index=True)
                N_unique = len(u)
                if N_unique > nt*nw:
                    print('sorting points by likelihood and re-starting from higest')
                    all_pts = all_pts[u,:]
                    all_logls = all_logls[u]

                    i = np.argsort(all_logls)[::-1]
                    N = nt*nw
                    pos0 = all_pts[i[:N],:].reshape((nt, nw, nd))
                else:
                    print(f'could not sort points by highest likelihood because only {N_unique} unique points')

                continue # Go around again, no convergence checks
            else:
                thin = 2*thin


            if conservative_convergence:
                # The chain is (Ntemp, Nwalker, Nstep, Ndim), so we take the cold chain with
                # `[0,...]` then average over the number of walkers, then put in a "singleton"
                # walker dimension for emcee, which wants (Nstep, Nwalker, Ndim) inputs.
                try:
                    avg_chain = np.mean(ptsampler.chain[0,...], axis=0)[:, None, :]
                    taus = emcee.autocorr.integrated_time(avg_chain)
                except emcee.autocorr.AutocorrError as er:
                    print('Looping with bad autocorrelation: ')
                    print(er)
                    continue
                converged = True
                break
            else:
                ess = az.ess(idata)
                min_ess = np.inf
                for k in ess.keys():
                    min_ess = min(min_ess, ess[k])
                if min_ess < converged_ess:
                    print(f'Looping with ess = {min_ess:.0f} < {converged_ess:.0f}')
                    continue
                else:
                    converged = True
                    break

    chain = {}
    for k in sampler.search_parameter_keys:
        chain[k] = idata.posterior[k].stack(dim=('chain', 'draw')).values
    chain_df = pd.DataFrame(chain, columns=sampler.search_parameter_keys)

    filled_chain = bilby.gw.conversion.generate_all_bbh_parameters(chain_df, likelihood=likelihood, priors=priors)

    nc = len(idata.posterior.chain)
    nd = len(idata.posterior.draw)

    cc = idata.posterior.chain.data
    dc = idata.posterior.draw.data


    for k, v in filled_chain.items():
        vv = v.values.reshape((nc, nd))
        idata.posterior[k] = (('chain', 'draw'), vv)

    # These are complex numbers and/or arbitrary Python objects, which NetCDF
    # won't store.  Sigh.
    del idata.posterior['waveform_approximant']
    del idata.posterior['H1_matched_filter_snr']
    del idata.posterior['L1_matched_filter_snr']

    az.to_netcdf(idata, op.join(outdir, 'posterior.nc'))

    