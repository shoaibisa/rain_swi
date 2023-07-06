import io
import math

from flask import Flask, jsonify, render_template, request, send_file, url_for
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


# making tif file into png
# def tif_to_img(tif_file,  type):
#     # Read the TIFF file using Pillow

#     image = Image.open(tif_file)

#     # Convert the image to a NumPy array
#     array = np.array(image)

#     # Create a Matplotlib figure and axes
#     fig, ax = plt.subplots()

#     # Set the color contrast (colormap)
#     cmap = plt.cm.plasma  # Change this to the desired colormap

#     # Get the minimum and maximum values from the array
#     if type == "clip":
#         min_value = 39
#     else:
#         min_value = -0.4
#     max_value = np.max(array)

#     # Normalize the data using the actual min and max values
#     norm = colors.Normalize(vmin=min_value, vmax=max_value)

#     # Apply the colormap to the plot with adjusted color range and normalization
#     im = ax.imshow(array, cmap=cmap, norm=norm)

#     # Set the background color as transparent
#     fig.patch.set_alpha(0.0)

#     cbar = plt.colorbar(im)

#     # Set custom ticks on the colorbar
#     cbar.set_ticks([min_value, max_value])

#     # Set corresponding tick labels
#     cbar.set_ticklabels([f"{min_value+0.4}", f"{max_value-1}"])

#     # Encode the plot image to base64 string
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png', transparent=True)
#     buffer.seek(0)
#     base64_image = base64.b64encode(buffer.read()).decode()

#     # Pass the base64 image to the template
#     return base64_image
def tif_to_img(tif_file, type):
    # Read the TIFF file using rasterio
    dataset = rasterio.open(tif_file)

    # Read the image data as a NumPy array
    array = dataset.read(1)

    # Get the spatial information from the TIFF file
    transform = dataset.transform
    min_x = transform * (0, 0)
    min_y = transform * (0, array.shape[0])
    max_x = transform * (array.shape[1], 0)
    max_y = transform * (array.shape[1], array.shape[0])

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots()

    # Set the color contrast (colormap)
    cmap = plt.cm.plasma  # Change this to the desired colormap

    # Get the minimum and maximum values from the array
    if type == "clip":
        min_value = 39
    else:
        min_value = -0.4
    max_value = np.max(array)

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
    cbar.set_ticklabels([f"{min_value+0.4}", f"{max_value-1}"])

    # Set the x-axis and y-axis labels with latitude and longitude information
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # set lat and long max min value in the plot
    # plt.xlim([min_x[0], max_x[0]])
    # plt.ylim([min_y[1], max_y[1]])

    # Adjust the positioning of the tick labels
    # plt.xticks(np.arange(min_x[0], max_x[0]))
    # plt.yticks(np.arange(min_y[1], max_y[1]), np.arange)
    # Set the DPI of the figure


    plt.xticks(np.arange(min_x[0], max_x[0], 5.1))
    plt.yticks(np.arange(min_y[1], max_y[1], 5.1))

    # Encode the plot image to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', transparent=True, dpi=1200)
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode()

    # Pass the base64 image to the template
    return base64_image


@app.route('/pixel_swi')
def pixel_swi():
    return render_template('pixel_swi.html')


@app.route('/pixel_swi', methods=['POST'])
def pixel_swi_post():

    # get lat long from input
    print(request.form)
    target_lat = float(request.form['lat'])
    target_lon = float(request.form['long'])
    # get the tif file from templates folder
    swi = []
    rain = []

    # Have pixel value of lat and long from each tif file in an array
    for i in range(1, 216):
        swi_file_path = os.path.join(
            'templates', 'uk_swi', 'uk_swi_'+str(i)+'.tif')
        rain_tiff_file_path = os.path.join(
            'templates', 'uk_rain', 'rainHourly_'+str(i)+'.tif')
        dataset_swi = rasterio.open(swi_file_path)
        dataset_rain = rasterio.open(rain_tiff_file_path)

        # Convert the target latitude and longitude to pixel coordinates
        col, row = dataset_swi.index(target_lon, target_lat)

        # Read the pixel value at the specified coordinates
        pixel_value_swi = dataset_swi.read(1)[row, col]
        pixel_value_rain = dataset_rain.read(1)[row, col]

        # Append the pixel value to the list
        swi.append(pixel_value_swi)
        rain.append(pixel_value_rain)

        # Close the TIFF file
        dataset_swi.close()
        dataset_rain.close()

    # Plot the graph with SWI as a red line and rainfall as blue bars
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(swi, color='red', label='SWI')
    ax1.bar(range(1, 216), rain, color='blue', label='Rainfall')
    ax1.set_xlabel('Time (hours)')
    ax2.set_ylabel('Soil Water Index (mm)')
    ax1.set_ylabel('Hourly rainfall (mm)')
    plt.title('SWI vs Rainfall')

    # Add legend to the graph
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    plt.savefig('templates/swi_vs_rain.png', dpi=1200)
    plt.close()

    # Convert the graph into base64-encoded string
    with open("templates/swi_vs_rain.png", "rb") as img_file:
        img_data = img_file.read()
        image_rain_swi = base64.b64encode(img_data).decode('utf-8')

    return render_template('swi_vs_rain.html', image_rain_swi=image_rain_swi)


@app.route('/calculate_swi')
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
    rain_tiff_file_path = os.path.join(
        'templates', 'uk_rain', 'rainHourly_'+str(num_tif)+'.tif')
    # tiff_file_path = os.path.join('templates', 'uk_swi', 'uk_swi_1.tif')

    try:
        clipped_tiff_file = clip_shape_zip(shape_zip_file, tiff_file_path)
    except ValueError as e:
        return render_template('error.html', error=str(e))

    # get image
    image_swi = tif_to_img(clipped_tiff_file, "clip")
    image_rain = tif_to_img(rain_tiff_file_path, "rain")

    return render_template('result.html', swi_image=image_swi, rain_image=image_rain, date=date, hour=hour, clipped_tiff_file=clipped_tiff_file)


@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    uploads_folder = app.config['UPLOAD_FOLDER']
    return send_file(os.path.join(uploads_folder, filename), as_attachment=True)


@app.route('/')
def index():
    return render_template('index.html')


# Custom filter to extract filename from file path
@app.template_filter('get_filename')
def get_filename(file_path):
    return os.path.basename(file_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4400)
