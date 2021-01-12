from pathlib import Path

import luigi
from threeML import FermiGBMBurstCatalog

from gbm_kitty.handlers.process import MakeSelections
from gbm_kitty.utils.configuration import gbm_kitty_config

base_path = Path(gbm_kitty_config["database"])


def build_catalog(n_per_chunk=4):

    cat = FermiGBMBurstCatalog()
    cat.query("t90>0")

    grbs = cat.result.index[:20]

    for grb in grbs:
        p: Path = base_path / grb

        if not p.exists():

            p.mkdir(parents=True)

    n_grbs = len(grbs)

    for i in range(0, n_grbs, n_per_chunk):
        this_slice = grbs[i:i+n_per_chunk]

        luigi.build([MakeSelections(grb_name=x) for x in this_slice],
                    workers=gbm_kitty_config["luigi"]["n_workers"], log_level="CRITICAL"

                    )
