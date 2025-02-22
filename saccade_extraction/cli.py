import click
from saccade_extraction.project import extractRealSaccades
from saccade_extraction.dlc import setConfigFile, analyzeVideosQuietly

@click.group()
def cli():
    pass

@cli.command()
@click.argument('config', type=str, required=True)
@click.argument('dlc', type=str, required=True)
@click.argument('ifi', type=str, required=True)
@click.option('--model', type=int, default=-1, help='Model index (default is -1)')
def extract(
    config,
    dlc,
    ifi,
    model
    ):
    """
    Extract real saccades

    \b
    CONFIG: Path to the saccade extraction config file
    DLC: Path to the DeepLabCut pose estimates
    IFI: Path to the file that stores inter-frame intervals
    """

    fileSets = [(
        dlc,
        ifi,
    ),]
    extractRealSaccades(
        config,
        fileSets,
        model
    )

    return

@cli.command()
@click.argument('config', type=str, required=True)
@click.argument('video', type=str, required=True)
def analyze(
    config,
    video
    ):
    """
    Analyze video and estimate pose

    \b
    CONFIG: Path to the DeepLabCut config file
    VIDEO: Path to the video to analyze
    """

    analyzeVideosQuietly(
        config,
        [video,],
        save_as_csv=True
    )

    return

if __name__ == '__main__':
    cli()