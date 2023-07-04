import io
import math

from flask import Flask, jsonify, render_template, request, send_file
import os
import geopandas as gpd
import rasterio
from matplotlib import colors
from rasterio.mask import mask
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import base64
from PIL import Image


import tempfile

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')


def clip_shape_zip(shape_zip_file, tif_file):
    """Clips a shape zip file with a tif file.

    Args:
        shape_zip_file (str): The path to the shape zip file.
        tif_file (str): The path to the tif file.

    Returns:
        str: The path to the clipped TIFF file.
    """

    with zipfile.ZipFile(shape_zip_file, "r") as zip_file:
        shape_files = [file for file in zip_file.namelist(
        ) if file.endswith(('.shp', '.shx', '.dbf', '.prj'))]

        if len(shape_files) == 0:
            raise ValueError("No valid shape files found in the shape zip.")

        # Extract shape files to a temporary folder
        temp_folder = 'temp_shape_files'
        os.makedirs(temp_folder, exist_ok=True)
        zip_file.extractall(temp_folder)

        # Load the shape file using geopandas
        shape_path = os.path.join(temp_folder, shape_files[0])
        shape_data = gpd.read_file(shape_path)

        # Read the TIFF file using rasterio
        with rasterio.open(tif_file) as tif:
            # Perform the clipping using the shape data
            clipped, _ = mask(tif, shape_data.geometry, crop=True)

            # Prepare the output TIFF file path
            output_folder = app.config['UPLOAD_FOLDER']
            os.makedirs(output_folder, exist_ok=True)
            output_tiff_file = os.path.join(output_folder, 'clipped_image.tif')

            # Write the clipped image as TIFF
            with rasterio.open(output_tiff_file, 'w', **tif.profile) as output:
                output.write(clipped)

    return output_tiff_file



@app.route('/')
def index():
    # Read the TIFF file using Pillow
    tif_file = "templates/uk_rain/rainHourly_100.tif"
    image = Image.open(tif_file)

    # Convert the image to a NumPy array
    array = np.array(image)

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots()

    # Set the color contrast (colormap)
    cmap = plt.cm.viridis  # Change this to the desired colormap

    # Get the minimum and maximum values from the array
    min_value = -1
    max_value = np.max(array)+1

    # Normalize the data using the actual min and max values
    norm = colors.Normalize(vmin=min_value, vmax=max_value)

    # Apply the colormap to the plot with adjusted color range and normalization
    im = ax.imshow(array, cmap=cmap, norm=norm)

    # Set the background color as transparent
    fig.patch.set_alpha(0.0)

    cbar = plt.colorbar(im)

    # Set custom ticks on the colorbar
    cbar.set_ticks([min_value, max_value])

    # Set corresponding tick labels
    cbar.set_ticklabels([f"{min_value+1}", f"{max_value-1}"])

    # Encode the plot image to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode()

    # Pass the base64 image to the template
    return render_template('index.html', plot_image=base64_image)


@app.route('/upload')
def upload_details():
    return render_template('upload.html')


@app.route('/calculate_swi', methods=['POST'])
def calculate_swi():
    shape_zip_file = request.files["shape_zip"]
    date = request.form["date"]
    hour = request.form["hour"]
    # get the date
    day = date.split('-')[2]
    num_tif = (int(day)-10)*24 + int(hour)+1
    tiff_file_path = os.path.join(
        'templates', 'uk_swi', 'uk_swi_'+str(num_tif)+'.tif')
    # tiff_file_path = os.path.join('templates', 'uk_swi', 'uk_swi_1.tif')

    try:
        clipped_tiff_file = clip_shape_zip(shape_zip_file, tiff_file_path)
    except ValueError as e:
        return render_template('error.html', error=str(e))

    return render_template('result.html', clipped_tiff_file=clipped_tiff_file)


@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    uploads_folder = app.config['UPLOAD_FOLDER']
    return send_file(os.path.join(uploads_folder, filename), as_attachment=True)


@app.route('/api/data')
def api():
    data = {
        'msg': 'hi',
        'val': 'uk'
    }
    return jsonify(data)


# Custom filter to extract filename from file path
@app.template_filter('get_filename')
def get_filename(file_path):
    return os.path.basename(file_path)


if __name__ == '__main__':
    app.run()


