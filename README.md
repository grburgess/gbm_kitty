<div  >
<img src="https://raw.githubusercontent.com/grburgess/gbm_kitty/master/logo.png" alt="drawing" width="500" align="right"/>
<header >
  <h1>
   <p style="color:#A233FF;"> GBM Kitty </p>
  </h1>
</header>

Database, reduce, and analyze GBM data without having to know anything. Curiosity killed the catalog. 

<br/>
</div>



## What is this?

* Creates a MongoDB database of GRBs observed by GBM. 
* Heuristic algorithms are applied to search for the background regions in the time series of GBM light curves. 
* Analysis notebooks can be generated on the fly for both time-instegrated and time-resolved spectral fitting. 

## What this is not

Animal cruelty. 

## What can you do?

Assuming you have built a local database (tis possible, see below), just type:

```bash
$> get_grb_analysis --grb GRBYYMMDDxxx

```

<img src="https://raw.githubusercontent.com/grburgess/gbm_kitty/master/media/nfit.gif" alt="drawing" width="800" align="center"/>


magic happens, and then you can look at your locally built GRB analysis notebook. 

If you want to do more, go ahead and fit the spectra:

```bash
$> get_grb_analysis --grb GRBYYMMDDxxx --run-fit

```

<img src="https://raw.githubusercontent.com/grburgess/gbm_kitty/master/media/fit.gif" alt="drawing" width="800" align="center"/>


And your automatic (but mutable) analysis is ready:

<img src="https://raw.githubusercontent.com/grburgess/gbm_kitty/master/media/nb.gif" alt="drawing" width="800" align="center"/>




## Building the database

The concept behind this is to query the Fermi GBM database for basic trigger info, use this in combination tools such as [gbmgeometry](https://gbmgeometry.readthedocs.io/en/latest/) to figure out which detectors produce the best data for each GRB, and then figure out preliminary selections / parameters / setups for subsequent analysis. 


```bash
$> build_catalog --n_grbs 100 --port 8989

```


This process starts with launching [luigi](https://luigi.readthedocs.io/en/stable/) which mangages the pipline:


<img src="https://raw.githubusercontent.com/grburgess/gbm_kitty/master/media/demo.png" alt="drawing" width="800" align="center"/>

All the of the metadata about the process is stored in a [mondoDB](https://www.mongodb.com) database which can be referenced later when building analyses.


