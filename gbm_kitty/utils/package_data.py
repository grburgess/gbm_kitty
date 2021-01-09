from pathlib import Path

from gbm_kitty.utils.configuration import gbm_kitty_config


def get_path_to_database() -> Path:

    p: Path = Path(gbm_kitty_config["database"])

    if not p.exists():

        p.mkdir(parents=True)

    return p
