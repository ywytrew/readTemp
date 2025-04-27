import time
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image,ImageTk
import glob
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from matplotlib.widgets import Slider
import pandas as pd
from tqdm import tqdm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy import io
import threading
import struct
import cv2

#required packages
#pip install matplotlib
#pip install numpy
#pip install pandas
#pip install scipy
#pip install tqdm

class imagereaderapp:
    def __init__(self, root):
        self.root = root
        self.file_path = None
        self.file_list = None
        self.total_frames = None
        self.allfield_temp = None
        self.current_frame = None
        self.Temp_list = None
        self.frame_size = None
        self.width = None
        self.height = None
        self.canvas_ver = None
        self.canvas_hor = None
        self.show_ver_enabled = False 
        self.show_hor_enabled = False

        self.ROI = None
        self.ROI_coords_x = tk.StringVar()
        self.ROI_coords_y = tk.StringVar()

        self.start_frame = None
        self.end_frame = None


        self.trackpoint_id = None
        self.trackpoint_coords = None

        # Current selection indicator
        self.current_marker = None
        self.current_line1 = None
        self.current_line2 = None

        # Define the temperature read area
        self.temperature_ROI = None
        self.temperature_hor = None
        self.temperature_ver = None
        
        # Define the size of ROI
        self.ROI_height = tk.StringVar()
        self.ROI_width = tk.StringVar()

        # Setup the Notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=0, column=0, sticky="nsew")  # Make the notebook expand in all directions

        # Configure the root grid to allow the notebook to expand
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        # Create the frames of the notebook
        self.image_frame = ttk.Frame(self.notebook)
        self.file_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.file_frame, text="DLtoTempeature")
        #self.notebook.add(self.image_frame, text="ROI")

        self.file_browser(self.file_frame)


    #the content display in the first slide
    def file_browser(self,root):
        self.open_button = tk.Button(root, text="Open File", command=self.open_bin)
        self.open_button.grid(row=1, column=0, padx=10, pady=10)

        #display file path
        self.convert_button = tk.Button(root,text="Convert DL to Temperature", command= self.convert_bin)
        self.convert_button.grid(row=2, column=0,padx=10, pady=10)

        self.status_label1 = tk.Label(root, text="")
        self.status_label1.grid(row=10,column=1)
        #self.status_label.pack(pady=10)
        self.progress1 = ttk.Progressbar(root, orient="horizontal", length=300, mode='determinate')
        self.progress1.grid(row=2, column=1)

        self.generate_preview = tk.Button(root,text="Generate Preview",command = self.Generate_Preview)
        self.generate_preview.grid(row=3, column=0)

        self.progress2 = ttk.Progressbar(root, orient="horizontal", length=300, mode='determinate')
        self.progress2.grid(row=3, column=1)
        
        self.refresh_button = tk.Button(root, text="Refresh", command=self.refresh)
        self.refresh_button.grid(row=4, column=0, padx=10, pady=10)

        return

    def open_bin(self):
        #clear the cache
        #Clear memory and temp files
        # Reset ROI data
        self.ROI = None
        self.ROI_coords_x.set("")
        self.ROI_coords_y.set("")
        self.ROI_height.set("")
        self.ROI_width.set("")
        self.start_frame = None
        self.end_frame = None
        
        # Clear markers and indicators if they exist
        if hasattr(self, 'canvas') and self.canvas is not None:
            if self.current_marker:
                self.canvas.delete(self.current_marker)
            if self.current_line1:
                self.canvas.delete(self.current_line1)
            if self.current_line2:
                self.canvas.delete(self.current_line2)
            self.canvas.delete("coordinate_text")
            
        # Reset indicators
        self.current_marker = None
        self.current_line1 = None
        self.current_line2 = None
        self.trackpoint_id = None
        
        # Reset canvas and displays
        self.show_ver_enabled = False
        self.show_hor_enabled = False
        if hasattr(self, 'canvas_ver') and self.canvas_ver is not None:
            self.canvas_ver.destroy()
            self.canvas_ver = None
        if hasattr(self, 'canvas_hor') and self.canvas_hor is not None:
            self.canvas_hor.destroy()
            self.canvas_hor = None
            
        # Force garbage collection to free memory
        gc.collect()



        #read all the .bin files in the path
        binary_path  = filedialog.askdirectory(title="Select the folder containing the .bin files")
        
        if binary_path:
            self.file_path = binary_path
            self.status_label1.config(text=binary_path)
            #read all the .bin files in the path
            self.file_list = sorted(glob.glob(os.path.join(self.file_path, "*.bin")))
            #Define the frame size
            self.width = 382
            self.height = 288
            self.frame_size = self.width * self.height

            self.total_frames = len(self.file_list)

            #create empty array to store the temperature data
            self.allfield_temp = np.zeros((self.height, self.width, self.total_frames))
            
        elif not binary_path:
            print("No folder selected")
        elif not os.path.exists(binary_path):
            print("The path does not exist")
        return
    
    def convert_bin(self):
        if self.file_path:
            try:
                self.progress1['maximum'] = self.total_frames
                
                # Pre-allocate memory more efficiently
                self.allfield_temp = np.zeros((self.height, self.width, self.total_frames), dtype=np.float32)
                
                # Process files in smaller batches to avoid memory issues
                batch_size = 500  # Process 500 frames at a time
                
                for batch_start in range(0, self.total_frames, batch_size):
                    batch_end = min(batch_start + batch_size, self.total_frames)
                    
                    for i in range(batch_start, batch_end):
                        self.progress1['value'] = i + 1
                        self.root.update()
                        
                        with open(self.file_list[i], 'rb') as file:
                            # Read timestamp (int64)
                            timestamp = struct.unpack('q', file.read(8))[0]
                            
                            # Use np.fromfile instead of frombuffer for better performance
                            DL_data = np.fromfile(file, dtype=np.int16, count=self.frame_size)/100
                            self.allfield_temp[:, :, i] = DL_data.reshape((self.height, self.width))
                    
                    # Force garbage collection after each batch
                    gc.collect()
                
                # Verify data was loaded properly
                max_temp = np.max(self.allfield_temp)
                min_temp = np.min(self.allfield_temp)
                print(f"Temperature range: {min_temp} to {max_temp}")
                
                # Save the data (corrected the variable name)
                np.save('Temperature_data', self.allfield_temp)
                self.status_label1.config(text=f"Conversion complete. Frames: {self.total_frames}")
                
            except MemoryError:
                self.status_label1.config(text="Memory error: Dataset too large")
            except Exception as e:
                self.status_label1.config(text=f"Error: {str(e)}")
                
        elif not self.file_path:
            print("No folder selected")
            self.status_label1.config(text="No folder selected")
        
        return

    #opencv is faster than matplotlib to save the image
    def Generate_Preview(self):
        if os.path.exists('Temperature_data.npy'):
            print('The temperature data detected! Begin to generate the preview.')

            if not os.path.exists('image'):
                os.makedirs('image')
            else:
                files = glob.glob('image\\*')
                for f in files:
                    os.remove(f)

            self.progress2['maximum'] = self.total_frames
            self.Temp_list = np.load('Temperature_data.npy')
            max_temp = np.max(self.Temp_list[:,:,0])
            min_temp = np.min(self.Temp_list[:,:,0])
            print('The maximum temperature is:', max_temp)
            print('The minimum temperature is:', min_temp)

            for i in range(self.total_frames):
                self.progress2['value'] = i
                self.root.update()

                frame = self.Temp_list[:,:,i]

                # Normalize to 0-255
                norm_frame = ((frame - min_temp) / (max_temp - min_temp) * 255).astype(np.uint8)

                # Apply colormap (JET)
                color_frame = cv2.applyColorMap(norm_frame, cv2.COLORMAP_JET)

                name = f'image\\{i}.png'
                cv2.imwrite(name, color_frame)

        else:
            print('Convert DL to temperature first!')
    '''
    def Generate_Preview(self):
        #check if the temperature data is exist
        if os.path.exists('Temperature_data.npy'):
            print('The temperature data dected!Begin to generate the preview')
            #self.status_label2.config(text='The temperature data is converted and saved!')
            #load the temperature data and save as image
            #create a new file to store the preview images
            #check if the folder is not exist
            if not os.path.exists('image'):
                os.makedirs('image')
            #check if the folder is exist, clear the folder and save new image in the folder
            elif os.path.exists('image'):
                #remove all the files in the folder
                files = glob.glob('image\\*')
                for f in files:
                    os.remove(f)
            #save the images
            self.progress2['maximum'] = self.total_frames
            self.Temp_list = np.load('Temperature_data.npy')
            max_temp = np.max(self.Temp_list[:,:,0])
            min_temp = np.min(self.Temp_list[:,:,0])
            print('The maximum temperature is:', max_temp)
            print('The minimum temperature is:',min_temp)
            for i in range(self.total_frames):
                self.progress2['value'] = i
                self.root.update()
                name = str(int(i)) + '.png'
                fig = plt.figure(figsize = (self.width/80,self.height/80), dpi = 80)
                #axes object with no margins
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.imshow(self.Temp_list[:,:,i], cmap='jet')
                plt.savefig('image\\' + name)
                plt.close()
        else:
            print('Convert DL to temperature first!')

        return
    '''

    def refresh(self):
        self.img_dir = 'image\\'
        #check if the temperature data npy file is exist, and the image folder is not empty
        if os.path.exists('Temperature_data.npy') and os.listdir(self.img_dir):
            self.Temp_list = np.load('Temperature_data.npy')
            self.image_files = sorted(glob.glob(os.path.join(self.img_dir,'*.png')), key=os.path.getmtime)
            self.canvas = tk.Canvas(self.root, width=382, height=288)
            self.total_frames = len(self.image_files)
            # Create event bindings for mouse dragging
            self.canvas.bind("<Button-1>", self.on_canvas_click)

            # Create a slider to select the frame
            self.slider_title = tk.Label(self.root, text="frame number")
            self.slider = tk.Scale(self.root,from_=0, to=len(self.image_files)-1, length=600,tickinterval=500, orient="horizontal", command=self.update_image)

            # Button frame
            button_frame = ttk.Frame(self.current_frame)
            button_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

            #create a button to apply the selection of ROI
            #self.ROI_apply = tk.Button(button_frame,text="Apply",command=self.apply_coordinates )

            # Create a button to select and print the ROI
            self.ROI_button = tk.Button(self.root, text="Select Coord", command=self.apply_coordinates)
            # Create a button to reset the ROI
            self.reset_button = tk.Button(self.root, text="Reset All", command=self.reset_All)

            # Create a dialog to display the selected ROI
            self.ROI_dialog_x = tk.Label(self.root, text="X:")
            self.ROI_text_x = ttk.Entry(self.root,textvariable=self.ROI_coords_x, width=3)
            self.ROI_dialog_y = tk.Label(self.root,text = "Y:")
            self.ROI_text_y = ttk.Entry(self.root,textvariable=self.ROI_coords_y, width =3)

            #Create a dialog to custume the size of ROI
            self.ROI_size_dialog_x = tk.Label(self.root,text="Width:")
            self.ROI_size_text_x = ttk.Entry(self.root,textvariable= self.ROI_width,width = 3)
            self.ROI_size_dialog_y = tk.Label(self.root,text = "Height:")
            self.ROI_size_text_y = ttk.Entry(self.root,textvariable= self.ROI_height,width = 3)

            self.ROI_size_select = tk.Button(self.root,text = "Select Size",command = self.select_size)

            # Create a button to mark the start frame and end frame
            self.start_frame = tk.Button(self.root, text="Select as Start", command=self.mark_start)
            self.end_frame = tk.Button(self.root, text="Select as End", command=self.mark_end)

            # Create a button to calculate the average temperature of the ROI
            self.temperature_ROI = tk.Button(self.root, text="ROI Temperature", command=self.read_temp)

            self.temperature_hor = tk.Button(self.root, text="Horizontal Temperature", command=self.read_temp_hon)
            self.temperature_ver = tk.Button(self.root, text="Vertical Temperature", command=self.read_temp_ver)

            self.show_ver = tk.Button(self.root, text="Show", command=self.show_ver_temp)
            self.show_hor = tk.Button(self.root, text="Show", command=self.show_hor_temp)

            #Layout
            self.update_image(0)
            self.canvas.grid(row=5, column=0, rowspan=20, columnspan=6)

            self.slider.grid(row=25, column=0,rowspan=1, columnspan=6)
            #self.slider_title.grid(row=7, column=0, columnspan=3)

            self.ROI_button.grid(row=8, column=7,sticky="w")
            self.reset_button.grid(row=10, column=7,sticky="w")

            self.ROI_dialog_x.grid(row=7, column=5, rowspan=1)
            self.ROI_text_x.grid(row=8, column=5,rowspan=1)

            self.ROI_dialog_y.grid(row=7, column=6,rowspan=1)
            self.ROI_text_y.grid(row=8, column=6,rowspan=1)

            self.ROI_size_dialog_x.grid(row=5, column=5,rowspan=1)
            self.ROI_size_text_x.grid(row=6, column=5,rowspan=1)

            self.ROI_size_dialog_y.grid(row=5, column=6,rowspan=1)
            self.ROI_size_text_y.grid(row=6, column=6,rowspan=1)

            self.ROI_size_select.grid(row=6, column=7)

            self.start_frame.grid(row=11, column=5,sticky="w")
            self.end_frame.grid(row=11, column=7,sticky="w")

            self.temperature_ROI.grid(row=12, column=5,sticky="w")

            self.temperature_ver.grid(row=13, column=5,sticky="w")
            self.temperature_hor.grid(row=14, column=5,sticky="w")
            self.show_ver.grid(row=13, column=7,sticky="w")
            self.show_hor.grid(row=14, column=7,sticky="w")
            #self.progress3 = ttk.Progressbar(self.root, orient="horizontal", length=100, mode='determinate')
            #self.progress3.grid(row=13, column=7)

        elif not os.path.exists('Temperature_data.npy'):
            self.status_label1.config(text="No temperature data, please convert the DL data first!")
        elif not os.listdir(self.img_dir):
            self.status_label1.config(text="No image data, please generate the preview first!")


    def update_image(self, value):
        index = int(value)
        image = Image.open(self.image_files[index])
        self.current_frame = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.current_frame)
        
        # Check if vertical display is enabled and canvas exists
        if self.show_ver_enabled and self.canvas_ver is not None:
            self.ver_path = 'Vertical_temp'
            ver_image_path = os.path.join(self.ver_path, f"{index}.png")
            
            # Check if the specific vertical image exists
            if os.path.exists(ver_image_path):
                image_ver = Image.open(ver_image_path)
                self.current_frame_ver = ImageTk.PhotoImage(image_ver)
                self.canvas_ver.delete("all")  # Clear canvas first
                self.canvas_ver.create_image(0, 0, anchor='nw', image=self.current_frame_ver)
            else:
                # Clear canvas if image doesn't exist
                self.canvas_ver.delete("all")
                self.canvas_ver.create_text(160, 120, text=f"No vertical data for frame {index}")
        else:
            pass
        #check if the horizontal temperature image is exist
        if self.show_hor_enabled and self.canvas_ver is not None:
            self.hor_path = 'Horizontal_temp'
            hor_image_path = os.path.join(self.hor_path, f"{index}.png")
            # Check if the specific horizontal image exists
            if os.path.exists(hor_image_path):
                image_hor = Image.open(hor_image_path)
                self.current_frame_hor = ImageTk.PhotoImage(image_hor)
                self.canvas_hor.delete("all")
                self.canvas_hor.create_image(0, 0, anchor='nw', image=self.current_frame_hor)
            else:
                # Clear canvas if image doesn't exist
                self.canvas_hor.delete("all")
                self.canvas_hor.create_text(160, 120, text=f"No horizontal data for frame {index}")
        else:
            pass

    def on_canvas_click(self, event):
        #set the trackpoint
        if self.trackpoint_id is not None:
            self.canvas.delete(self.trackpoint_id)
        x,y = event.x, event.y
        self.ROI_coords_x.set(x)
        self.ROI_coords_y.set(y)
        # Show marker on canvas
        self.show_marker(x, y)
    
    def apply_coordinates(self):
        if self.ROI_size_text_x.get() == "" or self.ROI_size_text_y.get() == "":
            tk.messagebox.showerror("Invalid Input", "Please enter width and height first")
        else:
            try:
                # Get coordinates from entry fields
                x = int(self.ROI_text_x.get())
                y = int(self.ROI_text_y.get())
                
                # Show marker on canvas
                self.show_marker(x, y)
                self.ROI_coords_x.set(x)
                self.ROI_coords_y.set(y)
            except ValueError:
                # Handle invalid input
                tk.messagebox.showerror("Invalid Input", "Please enter valid integer coordinates")
    
    def show_marker(self, x, y):
        # Remove previous marker if it exists
        if self.current_marker:
            self.canvas.delete(self.current_marker)
            self.canvas.delete(self.current_line1)
            self.canvas.delete(self.current_line2)
            self.canvas.delete("coordinate_text")
        # Draw new marker (crosshair)
        marker_size_x = int(self.ROI_width.get())
        marker_size_y = int(self.ROI_height.get())
        #print(type(marker_size_x),type(x))
        self.current_marker = self.canvas.create_oval(
            x - marker_size_x, y - marker_size_y,
            x + marker_size_x, y + marker_size_y,
            outline="red", width=2
        )
        #draw the vertical line and horizontal line cross whole image
        self.current_line1= self.canvas.create_line(0, y, 388, y, fill="red", width=1)
        self.current_line2 = self.canvas.create_line(x, 0, x, 286, fill="red", width=1)
        
        # Draw coordinate text
        self.canvas.create_text(
            x, y - marker_size_y-10,
            text=f"({x}, {y})",
            fill="red",
            tags="coordinate_text"
        )
        
    def reset_All(self):
        # Clear entry fields
        self.ROI_coords_x =  tk.StringVar()
        self.ROI_coords_y =  tk.StringVar()
        self.ROI_width = tk.StringVar()
        self.ROI_height = tk.StringVar()
        # Remove marker
        if self.current_marker:
            self.canvas.delete(self.current_marker)
            self.canvas.delete(self.current_line)
            self.canvas.delete("coordinate_text")
            self.current_marker = None
    
    def select_size(self):
        try:
            # Get size from entry fields
            width = int(self.ROI_size_text_x.get())
            height = int(self.ROI_size_text_y.get())
            
            # Set ROI size
            self.ROI_width.set(width)
            self.ROI_height.set(height)
            
        except ValueError:
            # Handle invalid input
            tk.messagebox.showerror("Invalid Input", "Please enter valid integer values for width and height")


    def mark_start(self):
        self.start_frame = self.slider.get()
        print(f"start frame: {self.start_frame}")
    
    def mark_end(self):
        self.end_frame = self.slider.get()
        print(f"end frame: {self.end_frame}")

    def read_temp(self):
        if self.start_frame is not None and self.end_frame is not None:
            if self.ROI_coords_x.get() != "" and self.ROI_coords_y.get() != "":
                if self.ROI_width.get() != "" and self.ROI_height.get() != "":
                    #calculate the average temperature of the ROI area for each frame
                    x,y = int(self.ROI_coords_x.get()), int(self.ROI_coords_y.get())
                    #define the ROI area
                    height = int(self.ROI_height.get())
                    width = int(self.ROI_width.get())
                    #print(type(height),type(y))
                    ROI = self.Temp_list[y-height:y+height, x-width:x+width, self.start_frame:self.end_frame]
                    #calculate the average temperature for each frame
                    avg_temp = np.mean(ROI, axis=(0,1))
                    print(f"Average temperature: {avg_temp}")

                    #save the temperature data as txt file
                    np.savetxt('Temperature_data.txt', avg_temp)

                    #plot the temperature data and save as png file
                    plt.figure()
                    plt.plot(avg_temp)
                    plt.title('Average Temperature')
                    plt.xlabel('Frame Number')
                    plt.ylabel('Temperature (C)')
                    #if the folder is not exist, create a new folder
                    if not os.path.exists('ROI'):
                        os.makedirs('ROI')
                    plt.savefig('ROI\\Temperature_ROI.png')
                    plt.close()

                    #pump up a new window to show the temperature fig
                    # Create a new window
                    temp_window = tk.Toplevel(self.root)
                    temp_window.title("Temperature Plot")
                    temp_window.geometry("600x400")
                    # Create a figure and axis
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(avg_temp)
                    ax.set_title('Average Temperature')

                    ax.set_xlabel('Frame Number')
                    ax.set_ylabel('Temperature (C)')
                    # Create a canvas to display the figure
                    canvas = FigureCanvasTkAgg(fig, master=temp_window)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    # Add a toolbar



                else:
                    tk.messagebox.showerror("Invalid Input","Please enter the width and height of the ROI")
            else:
                tk.messagebox.showerror("Invalid Input","Please enter the coordinates of the ROI")
        else:
            tk.messagebox.showerror("Invalid Input","Please select the start and end frame")

    #TODO
    '''
    #read the vertical tempreture at the ROI_coord
    def read_temp_ver(self):
        if self.start_frame is not None and self.end_frame is not None:
            if self.ROI_coords_x.get() != "" and self.ROI_coords_y.get() != "":
                if self.ROI_width.get() != "" and self.ROI_height.get() != "":
                    x = int(self.ROI_coords_y.get())
                    width = int(self.ROI_width.get())

                    # Read all vertical temp data
                    temp_vertical = self.Temp_list[:, x-width:x+width, self.start_frame:self.end_frame]
                    temp_vertical_avg = np.mean(temp_vertical, axis=1)

                    max_temp = np.max(temp_vertical_avg)
                    min_temp = np.min(temp_vertical_avg)

                    # Output folder
                    if not os.path.exists('Vertical_temp'):
                        os.makedirs('Vertical_temp')
                        #clear the folder
                    elif os.path.exists('Vertical_temp'):
                        files = glob.glob('Vertical_temp\\*')
                        for f in files:
                            os.remove(f)

                    img_width = 400
                    img_height = 300

                    for i in tqdm(range(self.start_frame, self.end_frame)):
                        # Create a blank image (black background)
                        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

                        values = temp_vertical_avg[:, i]
                        norm_values = ((values - min_temp) / (max_temp - min_temp))  # normalize to 0–1
                        y_vals = img_height - (norm_values * (img_height - 20)).astype(np.int32)
                        x_vals = np.linspace(10, img_width - 10, len(y_vals)).astype(np.int32)

                        # Draw polyline
                        points = np.array(list(zip(x_vals, y_vals)), np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [points], isClosed=False, color=(0, 0, 255), thickness=2)

                        # Add frame number
                        cv2.putText(img, f"Frame: {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                        cv2.putText(img, "Vertical Temp Profile", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        # Save image
                        cv2.imwrite(f'Vertical_temp\\{i}.png', img)

                else:
                    tk.messagebox.showerror("Invalid Input", "Please enter the width and height of the ROI")
            else:
                tk.messagebox.showerror("Invalid Input", "Please enter the coordinates of the ROI")
        else:
            tk.messagebox.showerror("Invalid Input", "Please select the start and end frame")
    '''
    def read_temp_ver(self):
        if self.start_frame is not None and self.end_frame is not None:
            if self.ROI_coords_x.get() != "" and self.ROI_coords_y.get() != "":
                if self.ROI_width.get() != "" and self.ROI_height.get() != "":
                    #calculate the average temperature of the ROI area for each frame
                    x = int(self.ROI_coords_y.get())
                    width = int(self.ROI_width.get())
                    #read all the vertical temperature data of all the frames
                    temp_vertical = self.Temp_list[:, x-width:x +width, self.start_frame:self.end_frame]
                    #calculate the average temperature of each row for each frame
                    temp_vertical_avg = np.mean(temp_vertical, axis=1)
                    max_temp = np.max(temp_vertical_avg)
                    min_temp = np.min(temp_vertical_avg)
                    #save the temperature data of every frame as png image, x axis is the vertical coordinate, y axis is the temperature
                    #check if the folder is not exist
                    if not os.path.exists('Vertical_temp'):
                        os.makedirs('Vertical_temp')
                    # Create a single figure and axis
                    fig, ax = plt.subplots(figsize=(10,8),dpi=300)
                
                    for i in tqdm(range(self.start_frame, self.end_frame)):
                        ax.clear()  # Clear previous plot
                        ax.plot(temp_vertical_avg[:,i])
                        ax.set_title(f'frame number: {i}')
                        ax.set_xlabel('Vertical Coordinate')
                        ax.set_ylabel('Temperature (C)')
                        ax.set_ylim(min_temp, max_temp)
                        fig.savefig(f'Vertical_temp\\{i}.png',)
                    
                    plt.close(fig)  # Close only once at the end


                #TODO
                else:
                    tk.messagebox.showerror("Invalid Input","Please enter the width and height of the ROI")
            else:
                tk.messagebox.showerror("Invalid Input","Please enter the coordinates of the ROI")
        else:
            tk.Messagebox.showerror("Invalid Input","Please select the start and end frame")


    def read_temp_hon(self):
        if self.start_frame is not None and self.end_frame is not None:
            if self.ROI_coords_x.get() != "" and self.ROI_coords_y.get() != "":
                if self.ROI_width.get() != "" and self.ROI_height.get() != "":
                    #calculate the average temperature of the ROI area for each frame
                    y = int(self.ROI_coords_x.get())
                    height = int(self.ROI_height.get())
                    #read all the horizontal temperature data of all the frames
                    temp_horizontal = self.Temp_list[y-height:y+height, :, self.start_frame:self.end_frame]
                    #calculate the average temperature of each column for each frame
                    temp_horizontal_avg = np.mean(temp_horizontal, axis=0)
                    max_temp = np.max(temp_horizontal_avg)
                    min_temp = np.min(temp_horizontal_avg)
                    #save the temperature data of every frame as png image, x axis is the vertical coordinate, y axis is the temperature
                    #check if the folder is not exist
                    if not os.path.exists('Horizontal_temp'):
                        os.makedirs('Horizontal_temp')
                    # Create a single figure and axis
                    fig, ax = plt.subplots(figsize=(10,8),dpi=300)
                    
                    for i in tqdm(range(self.start_frame, self.end_frame)):
                        ax.clear()
                        ax.plot(temp_horizontal_avg[:,i])
                        ax.set_title(f'frame number: {i}')
                        ax.set_xlabel('Horizontal Coordinate')
                        ax.set_ylabel('Temperature (C)')
                        ax.set_ylim(min_temp, max_temp)
                
                        fig.savefig(f'Horizontal_temp\\{i}.png')
                    plt.close(fig)
                else:
                    tk.messagebox.showerror("Invalid Input","Please enter the width and height of the ROI")
            else:
                tk.messagebox.showerror("Invalid Input","Please enter the coordinates of the ROI")
        else:
            tk.Messagebox.showerror("Invalid Input","Please select the start and end frame")

    #def read_temp_hor(self):

    #Customable ROI area, 

    def show_ver_temp(self):
        self.ver_path = "Vertical_temp"
        if os.path.exists(self.ver_path):
            # Check if the folder is not empty
            if os.listdir(self.ver_path):
                # Set flag to True
                self.show_ver_enabled = True
                self.canvas_ver = tk.Canvas(self.root, width=320, height=240)
                self.canvas_ver.grid(row=26, column=0, rowspan=6, columnspan=4)
                # Only create the canvas if it doesn't exist yet
                if self.canvas_ver is None:
                    self.canvas_ver = tk.Canvas(self.root, width=320, height=240)
                    self.canvas_ver.grid(row=26, column=0, rowspan=6, columnspan=4)
                
                # Force update of the current image
                self.update_image(self.slider.get())
            else:
                tk.messagebox.showerror("Error","Vertical temperature folder is empty")
        else:
            tk.messagebox.showerror("Error","Vertical temperature folder does not exist")

    def show_hor_temp(self):
        self.hor_path = "Horizontal_temp"
        if os.path.exists(self.hor_path):
            # Check if the folder is not empty
            if os.listdir(self.hor_path):
                # Set flag to True
                self.show_hor_enabled = True
                self.canvas_hor = tk.Canvas(self.root, width=320, height=240)
                self.canvas_hor.grid(row=26, column=5, rowspan=12, columnspan=8)
                # Only create the canvas if it doesn't exist yet
                if self.canvas_hor is None:
                    self.canvas_hor = tk.Canvas(self.root, width=320, height=240)
                    self.canvas_hor.grid(row=26, column=5, rowspan=12, columnspan=8)
                
                # Force update of the current image
                self.update_image(self.slider.get())
            else:
                tk.messagebox.showerror("Error", "Horizontal temperature folder is empty")
        else:
            tk.messagebox.showerror("Error", "Horizontal temperature folder does not exist")

def main():
    # 图像文件列表
    #img_dir = 'image//30-1'
    #temp_files = np.load("30-1.ptw.npy")
    #image_files = sorted(glob.glob(os.path.join(img_dir,'*.png')), key=os.path.getmtime)  # 示例文件名列表
    root = tk.Tk()
    app = imagereaderapp(root)
    #app.setup_layout()
    root.mainloop()

    return


if __name__ == '__main__':
    main()
