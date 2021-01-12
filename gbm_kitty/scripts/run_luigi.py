import click

from gbm_kitty.build import build_catalog
from gbm_kitty.database.create import build_primary_mongo_database


@click.command()
@click.option('--n_grbs', default=-1, help="build n_grbs", required=True)
def build_database_data(n_grbs):
    return build_catalog(n_grbs)

@click.command()
def build_database():
    return build_primary_mongo_database()


