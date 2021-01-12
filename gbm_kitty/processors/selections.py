import typing as tp

import astropy.io.fits as fits
import numpy as np
import ruptures as rpt
from joblib import Parallel, delayed
from threeML.utils.statistics.stats_tools import Significance
from threeML.utils.time_series.polynomial import polyfit
from threeML import update_logging_level
from gbm_kitty.utils.configuration import gbm_kitty_config


class AutoSelect(object):

    def __init__(self, *light_curves, verbose=False, max_time=200.):
        """


        """

        self._light_curves = light_curves
        self._n_light_curves: int = len(light_curves)

        self._verbose: bool = verbose

        self._max_time: float = max_time

    def process(self):
        """
        process the light curves
        """

        self._compute_primary_bkg_selection()
        self._clean_and_compute_changes()
        self._compute_significance()
        self._make_selections()

    @property
    def pre_time(self):
        return self._pre

    @property
    def post_time(self):
        return self._post

    @property
    def selections(self):
        return self._selections


    @property
    def brightest_det(self):
        return self._brightest_det
    
    def _compute_primary_bkg_selection(self):

        # number of sub divisions
        n_trials = 100

        # the end points of the background fit
        end_points = np.linspace(1, self._max_time, n_trials)

        log_likes = np.empty((self._n_light_curves, n_trials))

        # loop through each light curve

        for j, lc in enumerate(self._light_curves):

            # extract the light curve info

            _log_likes = np.empty(n_trials)

            y = lc.counts
            x = lc.mean_times
            exposure = lc.exposure

            # define a closure for this light curve

            def fit_backgrounds(i):

                update_logging_level("CRITICAL")
                # selections

                s1 = x < np.max([-i*.1, x.min() + 10])
                s2 = x > i

                idx = s1 | s2

                # fit background

                _, ll = polyfit(x[idx], y[idx], grade=3,
                                exposure=exposure[idx])

                return ll

            # continuosly increase the starting of the bkg
            # selections and store the log likelihood

            _log_likes = Parallel(n_jobs=8)(delayed(fit_backgrounds)(
                i) for i in end_points)

            log_likes[j, ...] = np.array(_log_likes)

        # now difference them all

        # the idea here is that the ordered log likes
        # will flatten out once a good background is
        # found and then we can identify that via its
        # change points

        delta = []
        for ll in log_likes:
            delta.append(np.diff(ll))

        delta = np.vstack(delta).T

        delta = delta.reshape(delta.shape[0], -1)

        delta = (delta - np.min(delta, axis=0).reshape((1, -1)) + 1)

        angles = angle_mapping(delta)

        dist = distance_mapping(delta)

        penalty = 2 * np.log(len(angles))
        algo = rpt.Pelt().fit(angles)
        cpts_seg = algo.predict(pen=penalty)

        # algo = rpt.Pelt().fit(dist)
        # cpts_seg2 = algo.predict(pen=penalty)

        algo = rpt.Pelt().fit(dist/dist.max())
        cpts_seg2 = algo.predict(pen=.01)

        tol = 1E-2
        best_range = len(dist)
        while (best_range >= len(dist)-1) and (tol < 1):
            for i in cpts_seg2:
                best_range = i

                if np.alltrue(np.abs(np.diff(dist/dist.max())[i:]) < tol):

                    break
                tol += 1e-2

        time = np.linspace(1, self._max_time, n_trials)[best_range+1]

        # now save all of this
        # and fit a polynomial to
        # each light curve and save it

        pre = np.max([-time*.1, x.min() + 10])
        post = time

        # create polys
        self._polys = []
        for j, lc in enumerate(self._light_curves):

            _log_likes = np.empty(n_trials)

            y = lc.counts
            x = lc.mean_times
            exposure = lc.exposure

            idx = (x < pre) | (x > post)

            p, ll = polyfit(x[idx], y[idx], grade=3, exposure=exposure[idx])

            self._polys.append(p)

        self._pre = pre
        self._post = post

    def _clean_and_compute_changes(self):

        tmp = []

        # now we go through and background substract the light curves

        for lc, poly in zip(self._light_curves, self._polys):

            n = len(lc.counts)

            bkg_counts = np.empty(n)
            bkg_errs = np.empty(n)

            for i, (a, b) in enumerate(zip(lc.time_bins[:-1], lc.time_bins[1:])):

                bkg_counts[i] = poly.integral(a, b)
                bkg_errs[i] = poly.integral_error(a, b)

            clean_counts = lc.counts - bkg_counts

            tmp.append(clean_counts)

        # look for the change points in the
        # in the cleaned light curves

        tmp = np.vstack(tmp).T

        angles = angle_mapping(tmp)

        penalty = 2 * np.log(len(angles))
        algo = rpt.Pelt().fit(angles)
        cpts_seg = algo.predict(pen=penalty)

        self._all_change_points = np.array(cpts_seg) - 1

    def _compute_significance(self):

        # compute the significance
        # between all the change points

        self._significance = []

        for lc, poly in zip(self._light_curves, self._polys):

            n = len(self._all_change_points) - 1

            bkg_counts = np.empty(n)
            bkg_errs = np.empty(n)

            time_bins = lc.time_bins[self._all_change_points]

            counts = []

            for a, b in zip(self._all_change_points[:-1], self._all_change_points[1:]):

                counts.append(lc.counts[a:b].sum())

            for i, (a, b) in enumerate(zip(time_bins[:-1], time_bins[1:])):

                bkg_counts[i] = poly.integral(a, b)
                bkg_errs[i] = poly.integral_error(a, b)

            sig = Significance(counts, bkg_counts)

            self._significance.append(
                sig.li_and_ma_equivalent_for_gaussian_background(bkg_errs))

    def _make_selections(self):

        # now make all the selections

        ii = 0
        max_sig = 0

        # find the brightest detector

        for i, sig in enumerate(self._significance):

            if sig.max() > max_sig:

                max_sig = sig.max()
                ii = i

        # grab all intervals with sigma greater
        # than 5

        self._brightest_det = ii
        
        idx = self._significance[ii] > gbm_kitty_config["selections"]["min_sig"]

        if idx.sum() == 0:

            self._selections = []

        else:

            select = slice_disjoint(np.where(idx)[0])

            self._selections = []

            for s1, s2 in select:

                t = self._light_curves[ii].mean_times[self._all_change_points]

                t1 = t[s1-1]
                t2 = t[s2-1]

                if t2-t1 > 1E-2:

                    self._selections.append([t1, t2])


def angle(x, ref_vector):
    """
    Calculate the separation angle between a vector and a reference vector
    """
    dot_prod = np.dot(x, ref_vector)

    norm1 = np.sqrt(np.sum(x ** 2))
    norm2 = np.sqrt(np.sum(ref_vector ** 2))

    if abs(dot_prod / (norm1 * norm2)) > 1:
        return 0

    else:
        return np.arccos(dot_prod / (norm1 * norm2))


def distance_mapping(x, ref_vector=None):
    """
    Maps a multi dimensional vector to the length of the vector
    """

    if ref_vector is None:
        ref_vector = np.repeat(1, x.shape[1])

    distance = np.sqrt(np.sum((x - ref_vector) ** 2, axis=1))

    return distance


def angle_mapping(x, ref_vector=None):
    """
    Map a multi demensional vector to the separation angle between the vector
    and a reference vector.
    If no reference vector is passed use the unity vector (1, 1, 1  ...)
    """
    if ref_vector is None:
        ref_vector = np.repeat(1, x.shape[1])

    x_ang = np.apply_along_axis(angle, axis=1, arr=x, ref_vector=ref_vector)

    return (x_ang / np.pi) * 360


class BinnedLightCurve(object):
    def __init__(self, counts, time_bins, tstart, tstop, dt, exposure=None):

        assert len(counts) == len(time_bins) - 1
        assert dt > 0

        self._counts = counts
        self._time_bins = time_bins
        self._dt = dt

        self._tstart = tstart
        self._tstop = tstop
        self._n_bins = len(counts)

        if exposure is None:
            self._exposure = self._time_bins[1:] - self._time_bins[:-1]
        else:

            self._exposure = exposure

    @property
    def counts(self):
        return self._counts

    @property
    def exposure(self):
        return self._exposure

    @property
    def mean_times(self):
        return 0.5 * (self._time_bins[:-1] + self._time_bins[1:])

    @property
    def time_bins(self):
        return self._time_bins

    @property
    def dt(self):
        return self._dt

    @property
    def tstart(self):
        return self._tstart

    @property
    def tstop(self):
        return self._tstop

    @property
    def n_bins(self):
        return int(self._n_bins)

    def time2idx(self, t):
        return np.searchsorted(self._time_bins, t)

    def get_src_counts(self, bkg_rate=500.0):

        return self._counts - bkg_rate * self._dt

    @classmethod
    def from_lightcurve(cls, lightcurve, tstart, tstop, dt):

        _, t, c = lightcurve.get_binned_light_curve(tstart, tstop, dt)

        return cls(c, t, tstart, tstop, dt)

    @classmethod
    def from_tte(cls, tte_file, tstart, tstop, dt, chan_start=64, chan_stop=96):

        tte = fits.open(tte_file)

        events = tte["EVENTS"].data["TIME"]
        pha = tte["EVENTS"].data["PHA"]

        # the GBM TTE data are not always sorted in TIME.
        # we will now do this for you. We should at some
        # point check with NASA if this is on purpose.

        # but first we must check that there are NO duplicated events
        # and then warn the user

        # sorting in time
        sort_idx = events.argsort()

        trigger_time = tte["PRIMARY"].header["TRIGTIME"]

        if not np.alltrue(events[sort_idx] == events):
            events = events[sort_idx]
            pha = pha[sort_idx]

        deadtime = np.zeros_like(events)
        overflow_mask = pha == 127  # specific to gbm! should work for CTTE

        # From Meegan et al. (2009)
        # Dead time for overflow (note, overflow sometimes changes)
        deadtime[overflow_mask] = 10.0e-6  # s

        # Normal dead time
        deadtime[~overflow_mask] = 2.0e-6  # s

        # apply mask

        events = events - trigger_time

        if events.min() > tstart:

            tstart = events.min()

        if tstop > events.max():

            tstop = events.max()

        bins = np.arange(tstart, tstop, dt)
        bins = np.append(bins, [bins[-1] + dt])

        dt2 = []
        counts = []

        s = (pha >= chan_start) & (pha <= chan_stop)

        for a, b in zip(bins[:-1], bins[1:]):

            mask = (events > a) & (events <= b)

            dt2.append(deadtime[mask].sum())

            mask = (events[s] > a) & (events[s] <= b)

            counts.append(mask.sum())

        exposure = np.ones(len(counts)) * dt - np.array(dt2)

        return cls(np.array(counts), np.array(bins), tstart, tstop, dt, exposure=exposure)


def slice_disjoint(arr):
    """
    Returns an array of disjoint indices from a bool array

    :param arr: and array of bools


    """

    slices = []
    start_slice = arr[0]
    counter = 0
    for i in range(len(arr) - 1):
        if arr[i + 1] > arr[i] + 1:
            end_slice = arr[i]
            slices.append([start_slice, end_slice])
            start_slice = arr[i + 1]
            counter += 1
    if counter == 0:
        return [[arr[0], arr[-1]]]
    if end_slice != arr[-1]:
        slices.append([start_slice, arr[-1]])
    return slices
