"""
Date: 06/09/2018
Version: 2.1
Description: Creates all the necessary directories
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""

####################### Importing libraries ####################################
import os
################################################################################

if __name__ == '__main__':
    # Creating directory if not there
    current_directory = os.getcwd()
    image_directory = current_directory + "/images"
    curves_directory = current_directory + "/plots"
    model_directory = current_directory + "/models"
    models_image_directory = current_directory + "/model_images"
    glove_directory = current_directory + "/glove_directory"

    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    if not os.path.exists(glove_directory):
        os.makedirs(glove_directory)

    if not os.path.exists(curves_directory):
        os.makedirs(curves_directory)

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    if not os.path.exists(models_image_directory):
        os.makedirs(models_image_directory)
