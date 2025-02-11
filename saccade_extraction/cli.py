import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument('folder', type=str)
def extract(folder):
    """
    Extract saccades
    """

    return

if __name__ == '__main__':
    cli()