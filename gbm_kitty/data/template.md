---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<img src="https://raw.githubusercontent.com/grburgess/gbm_kitty/master/logo.png" alt="drawing" width="500" align="left"/>
<div style="background-color:#A6A6A6" >

<header >
  <h1>
   <p style="color:#A233FF;"> GRB Analysis via GBM Kitty </p>
  </h1>


</header>


  <a target="blank" href="https://github.com/grburgess/gbm_kitty">See here for details</a>
  <p style="color:#A233FF;">This analysis was automatically generated to make your life easier.</p>
</div>



<!-- #region heading_collapsed=true -->
# Imports

<!-- #endregion -->

```python hidden=true
# Scientific libraries
import numpy as np


from threeML import *

import warnings
warnings.simplefilter("ignore")


import matplotlib.pyplot as plt
%matplotlib inline


silence_warnings()
update_logging_level("WARNING")

```

<!-- #region heading_collapsed=true -->
# Parameters
<!-- #endregion -->

```python tags=["parameters"] hidden=true

```

<!-- #region heading_collapsed=true -->
# Examining the catalog
<!-- #endregion -->

```python hidden=true
gbm_catalog = FermiGBMBurstCatalog()
gbm_catalog.query_sources(grb_name)
```

<!-- #region heading_collapsed=true -->
# Data Setup



<!-- #endregion -->

<!-- #region hidden=true -->
## Connect to the data
If the data is not present in the database, download it

<!-- #endregion -->

```python hidden=true
download = download_GBM_trigger_data(grb_trigger,
                                  detectors=gbm_detectors,
                                  destination_directory=download_dir)
```

<!-- #region hidden=true -->
## Setup the plugins
<!-- #endregion -->

```python hidden=true
fluence_plugins = []
time_series = {}

background_interval = [f"{float(bkg_pre - 20)}-{float(bkg_pre)}", f"{float(bkg_post)}-{float(bkg_post +50)}"]
source_interval = f"{float(src_start)}-{float(src_stop)}"



for det in gbm_detectors:

    
    ts_cspec = TimeSeriesBuilder.from_gbm_cspec_or_ctime(
        det,
        cspec_or_ctime_file=download[det]["cspec"],
        rsp_file=download[det]["rsp"], verbose=False
    )

    ts_cspec.set_background_interval(*background_interval)
    ts_cspec.save_background(f"{det}_bkg.h5", overwrite=True)

    ts_tte = TimeSeriesBuilder.from_gbm_tte(
        det,
        tte_file=download[det]["tte"],
        rsp_file=download[det]["rsp"],
        restore_background=f"{det}_bkg.h5",
        verbose=False
    )
    
    time_series[det] = ts_tte

    ts_tte.set_active_time_interval(source_interval)

    ts_tte.view_lightcurve(bkg_pre, bkg_post)
    
    fluence_plugin = ts_tte.to_spectrumlike()
    
    if det.startswith("b"):
        
        fluence_plugin.set_active_measurements("250-30000")
    
    else:
        
        fluence_plugin.set_active_measurements("9-900")
    
    fluence_plugin.rebin_on_background(1.)
    
    fluence_plugins.append(fluence_plugin)
```

```python hidden=true

brightest_ts = time_series[brightest_det]
brightest_ts.create_time_bins(src_start -5,
                              src_stop + 10, 
                              method="bayesblocks",
                              use_background=True, p0=0.1)

bad_bins = []
for i, w in enumerate(brightest_ts.bins.widths):
    
    if w < 5E-2:
        bad_bins.append(i)
    
    
edges = [brightest_ts.bins.starts[0]]

for i,b in enumerate(brightest_ts.bins):
    
    if i not in bad_bins:        
        edges.append(b.stop)

starts=edges[:-1]
stops = edges[1:]


brightest_ts.create_time_bins(starts, stops, method='custom')

n_intervals = len(starts)

time_resolved_plugins = {}

for k,v in time_series.items():
    v.read_bins(brightest_ts)
    
    v.view_lightcurve(use_binner=True);
    
    time_resolved_plugins[k] = v.to_spectrumlike(from_bins=True)
```

```python hidden=true

```

<!-- #region heading_collapsed=true -->
# Model
<!-- #endregion -->

<!-- #region hidden=true -->
## Band
<!-- #endregion -->

```python hidden=true
band = Band()
band.alpha.prior = Truncated_gaussian(lower_bound = -1.5, upper_bound = 1, mu=-1, sigma=0.5) 
band.beta.prior = Truncated_gaussian(lower_bound = -5, upper_bound = -1.6, mu=-2.25, sigma=0.5)
band.xp.prior = Log_normal(mu=np.log(1E2), sigma=1)
band.xp.bounds = (None, None)
band.K.prior = Log_uniform_prior(lower_bound = 1E-3, upper_bound = 1E1)
```

```python hidden=true
band_ps = PointSource(grb_name, ra=ra, dec=dec, spectral_shape=band)
band_model = Model(band_ps)
```

<!-- #region hidden=true -->
## CPL
<!-- #endregion -->

```python hidden=true
cpl = Cutoff_powerlaw()
cpl.index.prior = Truncated_gaussian(lower_bound = -1.5, upper_bound = 1, mu=-1, sigma=0.5) 
cpl.xc.prior = Log_normal(mu=np.log(1E2), sigma=1)
cpl.xc.bounds = (None, None)
cpl.K.prior = Log_uniform_prior(lower_bound = 1E-3, upper_bound = 1E1)
```

```python hidden=true
cpl_ps = PointSource(grb_name, ra=ra, dec=dec, spectral_shape=cpl)
cpl_model = Model(cpl_ps)
```

<!-- #region heading_collapsed=true -->
# Fit
<!-- #endregion -->

<!-- #region heading_collapsed=true hidden=true -->
## Fluence
<!-- #endregion -->

<!-- #region hidden=true -->
### Band Fit
<!-- #endregion -->

```python hidden=true
band_bayes = BayesianAnalysis(band_model, DataList(*fluence_plugins))
band_bayes.set_sampler("multinest")
band_bayes.sampler.setup(n_live_points=400, chain_name="chains/band_fit-")
```

```python hidden=true
if run_fits:
    band_bayes.sample()
    band_bayes.restore_median_fit()
    display_spectrum_model_counts(band_bayes, min_rate=20, step=False );
```

<!-- #region hidden=true -->
### CPL Fit
<!-- #endregion -->

```python hidden=true
cpl_bayes = BayesianAnalysis(cpl_model, DataList(*fluence_plugins))
cpl_bayes.set_sampler("multinest")
cpl_bayes.sampler.setup(n_live_points=400, chain_name="chains/cpl_fit-")
```

```python hidden=true
if run_fits:
    cpl_bayes.sample()
    cpl_bayes.restore_median_fit()
    display_spectrum_model_counts(cpl_bayes, min_rate=20, step=False );
```

<!-- #region hidden=true -->
### Compare
<!-- #endregion -->

```python hidden=true
if run_fits:
	fig = plot_point_source_spectra(band_bayes.results, cpl_bayes.results, flux_unit='erg2/(cm2 s keV)');
```

<!-- #region heading_collapsed=true hidden=true -->
## Time Resolved Analysis 
<!-- #endregion -->

<!-- #region heading_collapsed=true hidden=true -->
### Band Fit


<!-- #endregion -->

```python hidden=true
if run_fits:
    band_models = []
    band_results = []
    band_analysis = []
    for interval in range(n_intervals):

        # clone the model above so that we have a separate model
        # for each fit

        this_model = clone_model(band_model)

        # for each detector set up the plugin
        # for this time interval

        this_data_list = []
        for k, v in time_resolved_plugins.items():

            pi = v[interval]
            pi.remove_rebinning()
            
            if k.startswith("b"):
                pi.set_active_measurements("250-30000")
            else:
                pi.set_active_measurements("9-900")

            pi.rebin_on_background(1.0)

            this_data_list.append(pi)

        # create a data list

        dlist = DataList(*this_data_list)

        # set up the sampler and fit

        bayes = BayesianAnalysis(this_model, dlist)
        bayes.set_sampler("multinest")
        bayes.sampler.setup(n_live_points=500, chain_name=f"chains/band_fit_{interval}-")

        bayes.sample()

        # at this stage we could also
        # save the analysis result to
        # disk but we will simply hold
        # onto them in memory

        band_analysis.append(bayes)
```

<!-- #region heading_collapsed=true hidden=true -->
#### Examine the fits

<!-- #endregion -->

```python hidden=true
if run_fits:
    for a in band_analysis:
        a.restore_median_fit()
        display_spectrum_model_counts(a, step=False)
```
<!-- #region hidden=true -->


<!-- #endregion -->

```python hidden=true
if run_fits:
    plot_spectra(*[a.results for a in band_analysis[::1]],
                 flux_unit="erg2/(cm2 s keV)",
                 fit_cmap='viridis',
                 contour_cmap='viridis',
                 contour_style_kwargs=dict(alpha=0.1));
```
<!-- #region heading_collapsed=true hidden=true -->
### CPL Fit


<!-- #endregion -->

```python hidden=true
if run_fits:
    cpl_models = []
    cpl_results = []
    cpl_analysis = []
    for interval in range(n_intervals):

        # clone the model above so that we have a separate model
        # for each fit

        this_model = clone_model(cpl_model)

        # for each detector set up the plugin
        # for this time interval

        this_data_list = []
        for k, v in time_resolved_plugins.items():

            pi = v[interval]
            pi.remove_rebinning()

            if k.startswith("b"):
                pi.set_active_measurements("250-30000")
            else:
                pi.set_active_measurements("9-900")

            pi.rebin_on_background(1.0)

            this_data_list.append(pi)

        # create a data list

        dlist = DataList(*this_data_list)

        # set up the sampler and fit

        bayes = BayesianAnalysis(this_model, dlist)
        bayes.set_sampler("multinest")
        bayes.sampler.setup(n_live_points=500, chain_name=f"chains/cpl_fit_{interval}-")

        bayes.sample()

        # at this stage we could also
        # save the analysis result to
        # disk but we will simply hold
        # onto them in memory

        cpl_analysis.append(bayes)
```

<!-- #region hidden=true -->
#### Examine the fits

<!-- #endregion -->

```python hidden=true
if run_fits:
    for a in cpl_analysis:
        a.restore_median_fit()
        display_spectrum_model_counts(a, step=False)
```
<!-- #region hidden=true -->


<!-- #endregion -->

```python hidden=true
if run_fits:
    plot_spectra(*[a.results for a in cpl_analysis[::1]],
                 flux_unit="erg2/(cm2 s keV)",
                 fit_cmap='viridis',
                 contour_cmap='viridis',
                 contour_style_kwargs=dict(alpha=0.1));
```
<!-- #region hidden=true -->

<!-- #endregion -->
