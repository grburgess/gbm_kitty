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

Assuming you have built a local database (tis possible), just type:

```bash
$> get_grb_analysis --grb GRBYYMMDDxxx

```

magic happens, and then you can look at your locally built GRB analysis notebook. 

If you want to do more, go ahead and fit the spectra:

```bash
$> get_grb_analysis --grb GRBYYMMDDxxx --run-fit

```
