from pathlib import Path

import luigi
from threeML import FermiGBMBurstCatalog

from gbm_kitty.handlers.catalog import ScanCatalog
from gbm_kitty.utils.configuration import gbm_kitty_config

base_path = Path(gbm_kitty_config["database"])



def build_catalog(n_grbs=-1, port='8823' ):

    env_params = dict(scheduler_port=port,
                      scheduler_url=f"http://localhost:{port}/"

                      )

    
    luigi.build([ScanCatalog(n_grbs)],
                workers=gbm_kitty_config["luigi"]["n_workers"],
                log_level="CRITICAL",
                #no_lock=False
                **env_params
                )
