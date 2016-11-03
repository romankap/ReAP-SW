import os,sys
import datetime

output_file = None

ReLU_layer_name = "ReLU"
softmax_layer_name = "softmax"

################# AUX #################

def convert_number_if_needed(result, number_format=None):
    if number_format:
        return number_format.convert(result)
    return result

def convert_number_to_non_zero_if_needed(result, number_format=None):
    if number_format:
        return number_format.convert_to_non_zero(result)
    return result

def check_if_folder_exists_and_open(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def open_output_file(output_folder):
    check_if_folder_exists_and_open(output_folder)

    now = str(datetime.datetime.now()).replace(':','-').replace(' ','_')
    full_output_filename = output_folder + now + ".txt"
    global output_file
    output_file = open(full_output_filename, 'w')


def write_to_output_file(*strings_to_write):
    global output_file
    if output_file:
        print(*strings_to_write, file=output_file)
        output_file.flush()


def close_output_file():
    global output_file
    output_file.close()