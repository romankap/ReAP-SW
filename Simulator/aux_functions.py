import os,sys
import datetime

output_file = None

################# AUX #################
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
    print(*strings_to_write, file=output_file)
    output_file.flush()


def close_output_file():
    global output_file
    output_file.close()