import matplotlib as plt
import PySimpleGUI as sg
import os.path
import numpy as np
import open3d as o3d
from Reconstruction import Reconstruction
# First the window layout in 2 columns

menu_def = [['File', ['Open', 'Save', 'Exit', 'Properties']],
            ['Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
            ['Help', 'About...'], ]

file_list_column = [
    [ # Col 1 Row 1
        sg.Text("Select the pointcloud (.ply) for reconstruction: "),
        sg.In(size=(25, 1), disabled=True, enable_events=True, key="-INFILE-"),
        sg.FileBrowse(button_text="Browse",file_types=(".ply", ".*")),
    ],
    [
        sg.Text("Step Size: "),
        sg.Input(default_text="10", size=(10,10), key="-STEPSIZE-"),
        sg.Text("Window Length: "),
        sg.Input(default_text="60",size=(10,10), key="-WINDOWLEN-") #Remember to half the value of -WINDOWLEN-
    ],
    [ # Col 1 Row 1
        sg.Text("Name the output file: "),
        sg.Input(default_text="test_0.xyz", size=(25, 1), key="-OUTFILE-"),
        sg.Button(button_text="RUN", key="-RUN-", disabled=False),
    ],
    [ # Col 1 Row 2
        sg.Output(size=(50,10), key="-OUTPUT-", pad=((10,10),(0,0)))
    ],
    [
        sg.ProgressBar(100, size=(30, 8), key="-PROGBAR-")
    ],
]
image_viewer_column = [
    [ # Col 2 Row 1
        sg.Text("Menu", justification="center")
    ],
    [
        sg.Button(button_text="Capture", disabled=True),
        sg.Button(button_text="Save", disabled=True),
        sg.Button(button_text="Close", disabled=True)
    ],
    [ # Col 2 Row 2
        sg.Frame('Display', [[sg.Image("imgs/full_2_slices.png", size=(350, 200))]])
    ],
]

def progress(count, max):
    if count % int(max/20) != 0:
        return

    bar = "|"
    spaces = 20
    if count > int(max/20):
        for _ in range(count % int(max / 20)):
            bar += "-"
            spaces -= 1
    for _ in range(spaces):
        bar += " "
    bar += "|  " + str(count) + " / " + str(max)
    print(bar)

# ----- Full layout -----


layout = [
    [
        sg.Column(file_list_column, expand_x=True, expand_y=True),
        sg.VSeperator(),
        sg.Column(image_viewer_column, expand_y=True, expand_x=True, element_justification="center"),
    ],
]

window = sg.Window(title = "ITEMS: Volume Reconstruction", layout=layout, resizable=True)


in_file = ""
out_file = ""
step_size = 10
window_length = 60

# Run the Event Loop
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "EXIT"):
        break
    out_file = values["-OUTFILE-"]
    # Naming the input file
    #  -  Check if name is inputted and is of the correct file type.
    #  -  Print the input file name to the log.
    if event is "-INFILE-":
        in_file = values["-INFILE-"]
        if " " in in_file:
            print("Input filename is invalid. Do not include spaces in filename.")
            continue
        """
        TODO: add file checking
        """
        print("Input file: ", in_file.split('/')[-1])

    # Naming the output file
    #  -  Check whether output file is of correct file type.
    #  -  Disable RUN button if not given an output file name.
    #  -  Print the output file name to the log.
    if out_file is not "":
        out_file = values["-OUTFILE-"]
        if " " in out_file:
            print("Output filename is invalid. Do not include spaces in filename.")
            continue
        elif "." in out_file:
            out_file = out_file.split('.')[0] + ".xyz"
    else:
        window['-RUN-'].update(disabled=True)

    # Setting the step size & window length.
    if event is "-SETSIZE-":
        step_size = values["-STEPSIZE-"]
    if event is "-WINDOWLEN-":
        window_length = values["-WINDOWLEN-"]

    # Initialize reconstruction object and run reconstruction
    #       *Available only after the output file is given*
    #  -  Verify that input and output files have been established already.
    #  -  Update progress bar
    if event is "-RUN-" and in_file is not "" and out_file is not "":
        r = Reconstruction(in_file=in_file,out_file=out_file)

        recon_iter = r.reconstruct(slide=step_size, thickness=window_length/2)
        max_count = next(recon_iter)
        for count in recon_iter: # windowlen is twice thickness
            sg.OneLineProgressMeter(title="Reconstruction Progress", current_value=count, max_value=max_count, orientation='h')
window.close()