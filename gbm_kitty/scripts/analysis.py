import click

from gbm_kitty.kitty.generate_analysis import get_analysis_notebook



@click.command()
@click.option('--grb','-g', help="The GRB name: GRBYYMMDDXXX", required=True, type=str)
@click.option('--run-fit',   is_flag=True, help="Run the spectral fitting")
def get_grb_analysis(grb, run_fit):

    get_analysis_notebook(grb, destination=".",run_fit=run_fit)
