from pathlib import Path

import luigi
import numpy as np
import yaml
from gbmgeometry import (GBM, PositionInterpolator, download_trigdat,
                         get_official_location)
from threeML import download_GBM_trigger_data, silence_warnings

import gbm_kitty.handlers.process as process
from gbm_kitty.utils.configuration import gbm_kitty_config

from .grb_handler import GRB

silence_warnings()


base_path: Path = Path(gbm_kitty_config["database"])


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

        file_name = base_path / self.grb_name / \
            f"trigdat_bn{self.grb_name[3:]}.h5"

        return luigi.LocalTarget(file_name)

    def run(self):

        p: Path = base_path / self.grb_name

        file_name = Path(download_trigdat(
            self.grb_name, destination=p)).absolute()

        file_name_str = str(file_name)

        ver = file_name_str.split("_")[-1].split(".")[0]

        file_name_str = file_name_str.replace(f"_{ver}", "")

        file_name.rename(file_name_str)


class DownloadTriggerData(luigi.Task):

    grb_name = luigi.Parameter()

    def requires(self):

        return process.DetermineBestDetector(grb_name=self.grb_name)

    def output(self):

        p: Path = base_path / self.grb_name / "data_files.yml"

        return luigi.LocalTarget(p)

    def run(self):

        with self.input().open() as f:

            input = yaml.load(f, Loader=yaml.SafeLoader)

        files = download_GBM_trigger_data(f"bn{self.grb_name[3:]}",
                                          detectors=input["detector"],
                                          destination_directory=str(
                                              base_path/self.grb_name)


                                          )

        out = {}

        for k, v in files.items():
            tmp = {}
            for k1, v1 in v.items():
                tmp[k1] = str(v1)

            out[k] = tmp

        p: Path = base_path / self.grb_name / "data_files.yml"

        with p.open("w") as f:

            yaml.dump(data=out, stream=f, Dumper=yaml.SafeDumper)
