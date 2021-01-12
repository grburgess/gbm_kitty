from pathlib import Path

import mongoengine as moe
import yaml

from gbm_kitty.database.records import GRB, Selection
from gbm_kitty.utils.configuration import gbm_kitty_config

base_path = Path(gbm_kitty_config["database"])


def build_primary_mongo_database():

    moe.connect("gbmkitty")

    for grb_folder in base_path.glob("GRB*"):

        grb: moe.Document = GRB(name=str(grb_folder.name))

        # get the geometry

        with (grb_folder / "geometry.yml").open("r") as f:

            geometry = yaml.load(f, Loader=yaml.SafeLoader)

            location = geometry["location"]

            grb.location = [location["ra"], location["dec"], location["err"]]

            grb.detectors = geometry["detector"]

        # get the selections

        with (grb_folder / "selections.yml").open("r") as f:

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

        with (grb_folder / "data_files.yml").open("r") as f:

            data = yaml.load(f, Loader=yaml.SafeLoader)

            grb.data = data

        grb.save()

    moe.disconnect("gbmkitty")
