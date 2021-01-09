from pathlib import Path

import luigi

from gbm_kitty.utils.configuration import gbm_kitty_config


class GRB(luigi.Task):

    grb = luigi.Parameter()

    def requires(self):

        return None

    def output(self):

        p = Path(gbm_kitty_config["database"])

        return luigi.LocalTarget(p / self.grb / "grb_params.yml")
