from pathlib import Path

import luigi
import numpy as np
import yaml
from astropy.coordinates import SkyCoord
from gbmgeometry import (GBM, PositionInterpolator, download_trigdat,
                         get_official_location)

import gbm_kitty.handlers.download as download
from gbm_kitty.processors.selections import AutoSelect, BinnedLightCurve
from gbm_kitty.utils.configuration import gbm_kitty_config

base_path: Path = Path(gbm_kitty_config["database"])


class DetermineBestDetector(luigi.Task):

    grb_name = luigi.Parameter()

    def requires(self):

        return download.DownloadTrigdat(grb_name=self.grb_name)

    def output(self):

        file_name: Path = base_path / self.grb_name / "geometry.yml"

        return luigi.LocalTarget(file_name)

    def run(self):

        trigdat_file = base_path / self.grb_name / \
            f"trigdat_bn{self.grb_name[3:]}.h5"

        pi: PositionInterpolator = PositionInterpolator.from_trigdat_hdf5(
            trigdat_file)

        gbm: GBM = GBM.from_position_interpolator(pi)

        ra, dec, err = get_official_location(self.grb_name)

        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

        # get the separation

        separations = gbm.get_separation(coord)

        # get the Nais that are less
        # than 60 deg away

        nai = [f"n{i}" for i in range(10)]
        nai.extend(['na', 'nb'])

        nai = separations[nai]

        idx = nai < 60.

        out = nai[idx].sort_values()

        if len(out) < 2:

            idx = nai < 80.

            out = nai[idx].sort_values()

        dets = list(out.index[:3])

        # now get the closest BGO

        bgo = separations[["b0", "b1"]]

        idx = bgo.argmin()

        dets.append(bgo.index[idx])

        file_name: Path = base_path / self.grb_name / "geometry.yml"

        output = {}
        output["location"] = dict(ra=ra, dec=dec, err=err)
        output["detector"] = dets

        with file_name.open("w") as f:

            yaml.dump(data=output, stream=f, Dumper=yaml.SafeDumper)


class MakeSelections(luigi.Task):

    grb_name = luigi.Parameter()

    def requires(self):

        return download.DownloadTriggerData(grb_name=self.grb_name)

    def output(self):

        file_name = base_path / self.grb_name / "selections.yml"

        return luigi.LocalTarget(file_name)

    def run(self):

        with self.input().open() as f:

            input: dict = yaml.load(f, Loader=yaml.SafeLoader)

        # just get the NaIr

        tte_files = [v["tte"] for k, v in input.items() if k[0] != "b"]

        sub_config = gbm_kitty_config["selections"]

        light_curves = [BinnedLightCurve.from_tte(
            f, sub_config["tstart"], sub_config["tstop"], sub_config["dt"], 0, 128) for f in tte_files]

        auto_select = AutoSelect(
            *light_curves, max_time=sub_config["max_time"])

        auto_select.process()

        file_name: Path = base_path / self.grb_name / "selections.yml"

        output = {}
        output["background"] = dict(
            pre=float(np.round(auto_select.pre_time, 2)), post=float(np.round(auto_select.post_time, 2)))

        tmp = {}
        for i, selection in enumerate(auto_select.selections):

            a, b = selection

            tmp[i] = dict(start=float(a), stop=float(b))

        output["selections"] = tmp

        output["brightest_det"] = list(input.keys())[
            auto_select.brightest_det]

        with file_name.open("w") as f:

            yaml.dump(data=output, stream=f, Dumper=yaml.SafeDumper)
