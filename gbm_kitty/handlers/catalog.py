import re
from pathlib import Path

import luigi
import mongoengine as moe
import yaml
from threeML import FermiGBMBurstCatalog


from gbm_kitty.utils.configuration import gbm_kitty_config
import gbm_kitty.handlers.database as luigi_database
#from .download import DownloadTriggerData
#from .process import DetermineBestDetector, MakeSelections

base_path = Path(gbm_kitty_config["database"])


cat = FermiGBMBurstCatalog()
cat.query("t90>0")


class ScanCatalog(luigi.WrapperTask):

    max_grbs = luigi.IntParameter(-1)

   
    def requires(self):

        if self.max_grbs > 0:

            grbs = cat.result.index[:self.max_grbs]

        else:

            grbs = cat.result.index

        for grb in grbs:
            p: Path = base_path / grb

            if not p.exists():

                p.mkdir(parents=True)
        yield [luigi_database.UpdateDatabase(grb) for grb in grbs]
        # yield [MakeSelections(grb) for grb in grbs]
        # yield [CatalogSelections(grb) for grb in grbs]
        


class CatalogSelections(luigi.Task):

    grb_name = luigi.Parameter()

    def output(self):

        return base_path / self.grb_name / "official_selections.yml"

    def run(self):

        p = base_path / self.grb_name / "official_selections.yml"

        selections = cat.get_detector_information()[self.grb_name]

        selections["detectors"] = selections["detectors"].tolist()

        with p.open("w") as f:

            yaml.dump(stream=f, data=selections, Dumper=yaml.SafeDumper)


