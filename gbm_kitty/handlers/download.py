from pathlib import Path
import numpy as np
import yaml
import luigi
from astropy.coordinates import SkyCoord
from gbmgeometry import (GBM, PositionInterpolator, download_trigdat,
                         get_official_position)
from threeML import download_GBM_trigger_data

from gbm_kitty.utils.configuration import gbm_kitty_config

from .grb_handler import GRB


base_path: Path = Path(gbm_kitty_config["database"])


class DetermineBestDetector(luigi.Task):

    grb_name = luigi.Parameter()

    def requires(self):

        return DownloadTrigdat(grb_name=self.grb_name)

    def output(self):

        luigi.LocalTarget(self._file_name)

    def run(self):

        pi: PositionInterpolator = PositionInterpolator.from_trigdat_hdf5(self.input())

        gbm: GBM = GBM.from_position_interpolator(pi)

        ra, dec, err = get_official_position(self.grb_name)

        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

        # get the separation

        separations = gbm.get_separation(coord)

        # get the Nais that are less
        # than 60 deg away

        nai = [f"n{i}" for i in range(10)]
        nai.extend(['a','b'])

        nai = separations[nai]

        idx = nai < 60.

        dets = list(separations.index[idx])

        # now get the closest BGO

        bgo = separations[["b0", "b1"]]

        idx = bgo.argmin()

        dets.append("bgo")

        self._file_name: Path = base_path / self.grb_name / "geometry.yml"

        output = {}
        output["location"] = dict(ra=ra,dec=dec,err=err)
        output["detector"] = dets

        with self._file_name.open("w") as f:

            yaml.dump(f, Dumper=yaml.SafeDumper)
        

        

class DownloadTrigdat(luigi.Task):
    """
    Downloads a Trigdat file of a given
    version
    """
    priority = 100
    grb_name = luigi.Parameter()

    def requires(self):

        return GRB(grb=self.grb_name)

    def output(self):

        return luigi.LocalTarget(self._file_name)

    def run(self):

        p: Path = base_path / self.grb_name

        self._file_name = Path(download_trigdat(p)).absolute()


class DownloadTriggerData(luigi.Task):

    pass
