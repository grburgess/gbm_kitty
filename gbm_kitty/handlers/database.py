import re
from pathlib import Path

import luigi
import mongoengine as moe
import yaml

import gbm_kitty.handlers.catalog as luigi_cat
from gbm_kitty.database.records import GRB, OfficialSelection, Selection
from gbm_kitty.utils.configuration import gbm_kitty_config

from .download import DownloadTriggerData
from .process import DetermineBestDetector, MakeSelections

base_path = Path(gbm_kitty_config["database"])


matcher = re.compile('(-?\d*\.\d*)-(-?\d*\.\d*)')


class UpdateDatabase(luigi.Task):

    grb_name = luigi.Parameter()

    def output(self):

        p = base_path / self.grb_name / "hit_db"

        return luigi.LocalTarget(p)

    def requires(self):

        return dict(selection=MakeSelections(self.grb_name),
                    det=DetermineBestDetector(self.grb_name),
                    data=DownloadTriggerData(self.grb_name),
                    catalog=luigi_cat.CatalogSelections(self.grb_name)



                    )

    def run(self):

        moe.connect("gbmkitty")

        grb: moe.Document = GRB(name=self.grb_name)

        # get the geometry

        with self.input()["det"].open("r") as f:

            geometry = yaml.load(f, Loader=yaml.SafeLoader)

            location = geometry["location"]

            grb.location = [location["ra"], location["dec"], location["err"]]

            grb.detectors = geometry["detector"]

        # get the selections

        with self.input()["selection"].open("r") as f:

            selections = yaml.load(f, Loader=yaml.SafeLoader)

            bkg = selections["background"]

            grb.background_selection = [bkg["pre"], bkg["post"]]

            selection_record = Selection()

            _selections = selections["selections"]

            selection_record.start = [v["start"]
                                      for k, v in _selections.items()]

            selection_record.stop = [v["stop"] for k, v in _selections.items()]

            selection_record.save()

            grb.selection = selection_record

            grb.brightest_detector = selections["brightest_det"]

        # now get the data files

        with self.input()["data"].open("r") as f:

            data = yaml.load(f, Loader=yaml.SafeLoader)

            grb.data = data

        with self.input()["catalog"].open("r") as f:

            cat_data = yaml.load(f, Loader=yaml.SafeLoader)

            start, stop = matcher.match(
                cat_data["source"]["fluence"]).groups()

            cat_selection = Selection()

            cat_selection.start = [float(start)]
            cat_selection.stop = [float(stop)]

            cat_selection.save()

            official_selection = OfficialSelection(name=self.grb_name)
            official_selection.selection = cat_selection

            official_selection.detectors = cat_data["detectors"]

            official_selection.background_selection = cat_data["background"]
            official_selection.save()

        grb.save()

        moe.disconnect("gbmkitty")

        p = base_path / self.grb_name / "hit_db"

        p.touch()
