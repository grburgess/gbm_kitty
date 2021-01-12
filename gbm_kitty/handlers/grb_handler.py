from pathlib import Path

import luigi
import yaml

from gbm_kitty.processors.selections import AutoSelect, BinnedLightCurve
from gbm_kitty.utils.configuration import gbm_kitty_config

base_path: Path = Path(gbm_kitty_config["database"])


class GRB(luigi.Task):

    grb = luigi.Parameter()

    def requires(self):

        return None

    def output(self):

        file_name = base_path / self.grb / "grb_params.yml"

        return luigi.LocalTarget(file_name)

    def run(self):

        file_name: Path = base_path / self.grb / "grb_params.yml"

        file_name.touch()

