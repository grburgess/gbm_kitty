from pathlib import Path

import jupytext
import mongoengine
import papermill as pm

from gbm_kitty.database.records import GRB
from gbm_kitty.utils.configuration import gbm_kitty_config
from gbm_kitty.utils.package_data import get_path_to_template_analysis

base_path = Path(gbm_kitty_config["database"])

mongoengine.connect("gbmkitty")


def get_analysis_notebook(grb, destination=".", run_fit=False):

    destination = Path(destination)

    ntbk = jupytext.read(get_path_to_template_analysis())

    output: Path = destination / f"{grb}.ipynb"

    final_output = destination / f"{grb}_analysis.ipynb"
    
    jupytext.write(ntbk, output, fmt="ipynb")

    try:

        grb_db: GRB = GRB.objects(name=grb)[0]

    except:

        raise RuntimeError(f"{grb} is not in the database")

    parameters = {}

    parameters["grb_name"] = grb
    parameters["grb_trigger"] = f"bn{grb[3:]}"
    parameters["gbm_detectors"] = grb_db.detectors
    parameters["brightest_det"] = grb_db.brightest_detector
    parameters["bkg_pre"] = grb_db.background_selection[0]
    parameters["bkg_post"] = grb_db.background_selection[1]
    parameters["ra"] = grb_db.location[0]
    parameters["dec"] = grb_db.location[1]
    parameters["download_dir"] = str(base_path / grb)
    parameters["run_fits"] = run_fit


    start = min(grb_db.selection.start)
    stop = max(grb_db.selection.stop)

    parameters["src_start"] = start
    parameters["src_stop"] = stop
    
    pm.execute_notebook(
        output,
        final_output,
        parameters=parameters
    )

    output.unlink()
