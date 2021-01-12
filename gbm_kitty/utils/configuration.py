from configya import YAMLConfig

from pathlib import Path

structure = {}

structure["luigi"] = dict(n_workers=4)
structure["database"] = str(Path("~/.gbm_kitty/database").expanduser())
structure["selections"] = dict(tstart=-40, tstop=400, dt=0.25, min_sig=5, max_time=300)


class GBMKittyConfig(YAMLConfig):
    def __init__(self) -> None:

        super(GBMKittyConfig, self).__init__(
            structure=structure,
            config_path="~/.gbm_kitty",
            config_name="gbm_kitty_config.yml",
        )


gbm_kitty_config = GBMKittyConfig()
