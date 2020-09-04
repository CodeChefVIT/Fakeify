import click
from crop_face import tear_frame, crop_all_faces_in_folder


@click.command()
@click.option('-t', '--tear', required=True, help="mp4 to frames into train")
@click.option('-p', '--process', type=(int, int), required=True, default=(0, 0),
              help="(width, height) for resize cropped images in train folder")
def preprocess(tear, process):
    if tear != '/':
        click.echo(f'Tearing {tear}...')
        tear_frame(tear)
        click.echo('Tearing done.')

    click.echo()
    if process != (0, 0):
        click.echo(f'Cropping faces {process}...')
        crop_all_faces_in_folder(width=process[0], height=process[1])
        click.echo('Done Cropping.\n')

    click.echo('Happy Model Training :)')


if __name__ == '__main__':
    preprocess()
