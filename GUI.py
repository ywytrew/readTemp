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
from threading import Thread
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
        #self.ROI_coord_x = tk.StringVar()
        #self.ROI_coord_y = tk.StringVar()
        self.ROI_coords_x = tk.StringVar()
        self.ROI_coords_y = tk.StringVar()
        self.x_select = None
        self.y_select = None

        self.start_frame = None
        self.end_frame = None

        self.path = None
        self.img_dir = None

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
            #print(self.file_list)
            #Define the frame size
            self.width = 382
            self.height = 288
            self.frame_size = self.width * self.height

            self.total_frames = len(self.file_list)

            #create empty array to store the temperature data
            self.allfield_temp = np.zeros((self.height, self.width, self.total_frames))
        
            self.folder_name = os.path.basename(binary_path)
            self.path =  'image' + '\\' + self.folder_name
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
                
                # Create the output directory if it doesn't exist
                if not os.path.exists(self.path+'\\'+'Preview'):
                    os.makedirs(self.path+'\\'+'Preview')
                elif os.path.exists(self.path+'\\'+'Preview'):
                    #remove all the files in the folder
                    files = glob.glob(self.path +'\\'+'Preview' + '\\*')
                    for f in files:
                        os.remove(f)

                # Save the data (corrected the variable name)
                np.save(self.path +'\\'+'Temperature_ROI', self.allfield_temp)
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
        if os.path.exists(self.path +'\\'+'Temperature_ROI.npy'):
            print('The temperature data detected! Begin to generate the preview.')

            self.progress2['maximum'] = self.total_frames
            self.Temp_list = np.load(self.path + '\\'+'Temperature_ROI.npy')
            max_temp = np.max(self.Temp_list[:,:,0])
            min_temp = np.min(self.Temp_list[:,:,0])
            print('The maximum temperature is:', max_temp)
            print('The minimum temperature is:', min_temp)

            path  = self.path + '\\' + 'Preview'

            for i in range(self.total_frames):
                self.progress2['value'] = i
                self.root.update()

                frame = self.Temp_list[:,:,i]

                # Normalize to 0-255
                norm_frame = ((frame - min_temp) / (max_temp - min_temp) * 255).astype(np.uint8)

                # Apply colormap (JET)
                color_frame = cv2.applyColorMap(norm_frame, cv2.COLORMAP_JET)

                name = f'{path}\\{i}.png'
                cv2.imwrite(name, color_frame)

            
            self.status_label1.config(text='Preview images generation complete!')

        else:
            print('Convert DL to temperature first!')

    def refresh(self):
        # Reset previous UI elements
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Canvas) or isinstance(widget, tk.Scale) or isinstance(widget, tk.Button) or isinstance(widget, tk.Label) or isinstance(widget, ttk.Entry):
                widget.destroy()

        self.img_dir = self.path + '\\Preview'+'\\'
        
        # Check if temperature data and preview images exist
        if os.path.exists(self.path + '\\'+'Temperature_ROI.npy') and os.listdir(self.img_dir):
            try:
                self.Temp_list = np.load(self.path + '\\'+'Temperature_ROI.npy')
                self.image_files = sorted(glob.glob(os.path.join(self.img_dir,'*.png')), key=os.path.getmtime)
                
                # Create a new canvas with a fixed reference to prevent garbage collection
                self.canvas = tk.Canvas(self.root, width=382, height=288)
                self.total_frames = len(self.image_files)
                
                # Recreate the slider
                self.slider = tk.Scale(self.root, from_=0, to=len(self.image_files)-1, 
                                    length=600, tickinterval=500, 
                                    orient="horizontal", command=self.update_image)
                
                # Recreate other UI elements
                self.ROI_button = tk.Button(self.root, text="Select Coord", command=self.apply_coordinates)
                self.reset_button = tk.Button(self.root, text="Reset All", command=self.reset_All)
                
                self.ROI_dialog_x = tk.Label(self.root, text="X:")
                self.ROI_text_x = ttk.Entry(self.root, textvariable=self.ROI_coords_x, width=3)
                self.ROI_dialog_y = tk.Label(self.root, text="Y:")
                self.ROI_text_y = ttk.Entry(self.root, textvariable=self.ROI_coords_y, width=3)
                
                self.ROI_size_dialog_x = tk.Label(self.root, text="Width:")
                self.ROI_size_text_x = ttk.Entry(self.root, textvariable=self.ROI_width, width=3)
                self.ROI_size_dialog_y = tk.Label(self.root, text="Height:")
                self.ROI_size_text_y = ttk.Entry(self.root, textvariable=self.ROI_height, width=3)
                
                self.ROI_size_select = tk.Button(self.root, text="Select Size", command=self.select_size)
                
                self.start_frame = tk.Button(self.root, text="Select as Start", command=self.mark_start)
                self.end_frame = tk.Button(self.root, text="Select as End", command=self.mark_end)
                
                self.temperature_ROI = tk.Button(self.root, text="ROI Temperature", command=self.read_temp)
                self.temperature_hor = tk.Button(self.root, text="Horizontal Temperature", command=self.read_temp_hon)
                self.temperature_ver = tk.Button(self.root, text="Vertical Temperature", command=self.read_temp_ver)
                
                self.show_ver = tk.Button(self.root, text="Show", command=self.show_ver_temp)
                self.show_hor = tk.Button(self.root, text="Show", command=self.show_hor_temp)
                
                # Create the first image
                self.current_frame = None
                
                # Layout
                self.canvas.grid(row=5, column=0, rowspan=20, columnspan=6)
                self.slider.grid(row=25, column=0, rowspan=1, columnspan=6)
                
                self.ROI_button.grid(row=8, column=7, sticky="w")
                self.reset_button.grid(row=10, column=7, sticky="w")
                
                self.ROI_dialog_x.grid(row=7, column=5, rowspan=1)
                self.ROI_text_x.grid(row=8, column=5, rowspan=1)
                
                self.ROI_dialog_y.grid(row=7, column=6, rowspan=1)
                self.ROI_text_y.grid(row=8, column=6, rowspan=1)
                
                self.ROI_size_dialog_x.grid(row=5, column=5, rowspan=1)
                self.ROI_size_text_x.grid(row=6, column=5, rowspan=1)
                
                self.ROI_size_dialog_y.grid(row=5, column=6, rowspan=1)
                self.ROI_size_text_y.grid(row=6, column=6, rowspan=1)
                
                self.ROI_size_select.grid(row=6, column=7)
                
                self.start_frame.grid(row=11, column=5, sticky="w")
                self.end_frame.grid(row=11, column=7, sticky="w")
                
                self.temperature_ROI.grid(row=12, column=5, sticky="w")
                
                self.temperature_ver.grid(row=13, column=5, sticky="w")
                self.temperature_hor.grid(row=14, column=5, sticky="w")
                self.show_ver.grid(row=13, column=7, sticky="w")
                self.show_hor.grid(row=14, column=7, sticky="w")
                
                self.status_label2 = tk.Label(self.root, text="")
                self.status_label2.grid(row=13, column=6, columnspan=1, sticky="w")
                self.status_label3 = tk.Label(self.root, text="")
                self.status_label3.grid(row=14, column=6, columnspan=1, sticky="w")


                # Bind canvas click event
                self.canvas.bind("<Button-1>", self.on_canvas_click)
                
                # Update image to the first frame
                self.update_image(0)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to refresh: {str(e)}")
        else:
            messagebox.showinfo("Info", "No temperature data or preview images available. Please convert DL data and generate preview first.")
    
    def update_image(self, value):
        index = int(value)
        image = Image.open(self.image_files[index])
        self.current_frame = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.current_frame)
        
        # Update vertical temperature image if enabled
        if hasattr(self, 'show_ver_enabled') and self.show_ver_enabled and hasattr(self, 'canvas_ver') and self.canvas_ver is not None:
            self.ver_path = self.path + '\\' + "Vertical_temp"
            ver_image_path = os.path.join(self.ver_path, f"{index}.jpg")
            
            # Check if the specific vertical image exists
            if os.path.exists(ver_image_path):
                image_ver = Image.open(ver_image_path)
                self.current_frame_ver = ImageTk.PhotoImage(image_ver)
                self.canvas_ver.delete("all")  # Clear canvas first
                self.canvas_ver.create_image(0, 0, anchor='nw', image=self.current_frame_ver)
                # Optionally add frame number indicator
                self.canvas_ver.create_text(80, 20, text=f"Frame: {index}", fill="white", 
                                        font=("Arial", 12), anchor='w')
            else:
                # Clear canvas if image doesn't exist
                self.canvas_ver.delete("all")
                self.canvas_ver.create_text(300, 200, text=f"No vertical data for frame {index}", 
                                        font=("Arial", 12))
    
        else:
            pass
        #check if the horizontal temperature image is exist
        if hasattr(self, 'show_hor_enabled') and self.show_hor_enabled and hasattr(self, 'canvas_hor') and self.canvas_hor is not None:
            self.hor_path = self.path + '\\' +"Horizontal_temp"
            hor_image_path = os.path.join(self.hor_path, f"{index}.jpg")
            
            # Check if the specific horizontal image exists
            if os.path.exists(hor_image_path):
                image_hor = Image.open(hor_image_path)
                self.current_frame_hor = ImageTk.PhotoImage(image_hor)
                self.canvas_hor.delete("all")
                self.canvas_hor.create_image(0, 0, anchor='nw', image=self.current_frame_hor)
                # Optionally add frame number indicator
                self.canvas_hor.create_text(80, 20, text=f"Frame: {index}", fill="white", 
                                        font=("Arial", 12), anchor='w')
            else:
                # Clear canvas if image doesn't exist
                self.canvas_hor.delete("all")
                self.canvas_hor.create_text(300, 200, text=f"No horizontal data for frame {index}", 
                                        font=("Arial", 12))
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
                self.x_select = x
                self.y_select = y
            except ValueError:
                # Handle invalid input
                tk.messagebox.showerror("Invalid Input", "Please enter valid integer coordinates")
            
            #print(f"Selected coordinates: ({x}, {y})")
            print(f"Selected coordinates: ({self.x_select}, {self.y_select})")
    
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
        self.ROI_coords_x.set("")
        self.ROI_coords_y.set("")
        self.ROI_width.set("")
        self.ROI_height.set("")
        #self.x_select = None
        #self.y_select = None
        # Remove marker
        self.x_select = None
        self.y_select = None
        self.start_frame = None
        self.end_frame = None
        if self.current_marker:
            self.canvas.delete(self.current_marker)
            self.canvas.delete(self.current_line1)
            self.canvas.delete(self.current_line2)
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
            if self.x_select is not None and self.y_select is not None:
                if self.ROI_width.get() != "" and self.ROI_height.get() != "":
                    #calculate the average temperature of the ROI area for each frame
                    x,y = int(self.x_select), int(self.y_select)
                    #define the ROI area
                    height = int(self.ROI_height.get())
                    width = int(self.ROI_width.get())
                    #print(type(height),type(y))
                    ROI = self.Temp_list[y-height:y+height, x-width:x+width, self.start_frame:self.end_frame]
                    #calculate the average temperature for each frame
                    avg_temp = np.mean(ROI, axis=(0,1))
                    print(f"Average temperature: {avg_temp}")

                    #save the temperature data as txt file
                    np.savetxt(self.path +'\\'+'Temperature_data.txt', avg_temp)

                    #plot the temperature data and save as png file
                    plt.figure()
                    plt.plot(avg_temp)
                    plt.title('Average Temperature')
                    plt.xlabel('Frame Number')
                    plt.ylabel('Temperature (C)')
                    #if the folder is not exist, create a new folder
                    if not os.path.exists('ROI'):
                        os.makedirs('ROI')
                    plt.savefig(self.path +'\\'+'Temperature_ROI.png')
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
                tk.messagebox.showerror("Invalid Input","Please enter the coordinates of the ROI and click select coord")
        else:
            tk.messagebox.showerror("Invalid Input","Please select the start and end frame")

    def read_temp_ver(self):
        if self.start_frame is not None and self.end_frame is not None:
            if self.x_select != "" and self.y_select != "":
                if self.ROI_width.get() != "" and self.ROI_height.get() != "":
                    Y_axis = np.arange(0, self.height, 1)

                    #calculate the average temperature of the ROI area for each frame
                    x = int(self.x_select)
                    width = int(self.ROI_width.get())
                    #read all the vertical temperature data of all the frames
                    temp_vertical = self.Temp_list[:, x-width:x +width, self.start_frame:self.end_frame]
                    #calculate the average temperature of each row for each frame
                    temp_vertical_avg = np.mean(temp_vertical, axis=1)
                    max_temp = np.max(temp_vertical_avg)
                    min_temp = np.min(temp_vertical_avg)
                    #save the temperature data of every frame as png image, x axis is the vertical coordinate, y axis is the temperature
                    #check if the folder is not exist
                    if not os.path.exists(self.path + '\\' +'Vertical_temp'):
                        os.makedirs(self.path + '\\' +'Vertical_temp')
                    #check if the folder is exist, clear the folder and save new image in the folder
                    elif os.path.exists(self.path + '\\' +'Vertical_temp'):
                        #remove all the files in the folder
                        files = glob.glob(self.path + '\\' +'Vertical_temp\\*')
                        for f in files:
                            os.remove(f)

                    # Create a single figure and axis
                    fig, ax = plt.subplots(figsize=(8,6),dpi=100)
                    path = self.path + '\\' +'Vertical_temp'
                    ax.set_xlabel('Vertical Coordinate')
                    ax.set_ylabel('Temperature (C)')
                    ax.set_ylim(min_temp, max_temp)

                    for i in tqdm(range(self.start_frame, self.end_frame)):
                        # Clear previous plot
                        ax.clear()
                        ax.scatter(Y_axis,temp_vertical_avg[:,i])
                        ax.set_title(f'frame number: {i}')
                        fig.savefig(f'{path}\\{i}.jpg',
                                    bbox_inches = None)

                        self.status_label2.config(text = f"{i+1}/ {self.end_frame}")
                        self.root.update()
                    plt.close(fig)  # Close only once at the end

                    #save the all temperature data as npy file
                    np.save(self.path + '\\' +'Vertical_temp.npy', temp_vertical_avg)

                #TODO
                else:
                    tk.messagebox.showerror("Invalid Input","Please enter the width and height of the ROI")
            else:
                tk.messagebox.showerror("Invalid Input","Please enter the coordinates of the ROI")
        else:
            tk.Messagebox.showerror("Invalid Input","Please select the start and end frame")


    def read_temp_hon(self):
        if self.start_frame is not None and self.end_frame is not None:
            if self.x_select != "" and self.y_select != "":
                if self.ROI_width.get() != "" and self.ROI_height.get() != "":
                    #create X axis, pixel number is the self.width of the image 
                    X_axis = np.arange(0, self.width, 1)
                    #calculate the average temperature of the ROI area for each frame
                    y = int(self.y_select)
                    height = int(self.ROI_height.get())
                    #read all the horizontal temperature data of all the frames
                    temp_horizontal = self.Temp_list[y-height:y+height, :, self.start_frame:self.end_frame]
                    #calculate the average temperature of each column for each frame
                    temp_horizontal_avg = np.mean(temp_horizontal, axis=0)
                    max_temp = np.max(temp_horizontal_avg)
                    min_temp = np.min(temp_horizontal_avg)
                    #save the temperature data of every frame as png image, x axis is the vertical coordinate, y axis is the temperature
                    #check if the folder is not exist
                    if not os.path.exists(self.path + '\\' + 'Horizontal_temp'):
                        os.makedirs(self.path + '\\' + 'Horizontal_temp')
                    #check if the folder is exist, clear the folder and save new image in the folder
                    elif os.path.exists(self.path + '\\' +'Horizontal_temp'):
                        #remove all the files in the folder
                        files = glob.glob(self.path + '\\' +'Horizontal_temp\\*')
                        for f in files:
                            os.remove(f)
                    # Create a single figure and axis
                    fig, ax = plt.subplots(figsize=(8,6),dpi=100)
                    ax.set_xlabel('Horizontal Coordinate')
                    ax.set_ylabel('Temperature (C)')
                    path = self.path + '\\' +'Horizontal_temp'
                    for i in tqdm(range(self.start_frame, self.end_frame)):
                        ax.clear()
                        ax.scatter(X_axis,temp_horizontal_avg[:,i])
                        ax.set_title(f'frame number: {i}')
                        ax.set_ylim(min_temp, max_temp)
                
                        fig.savefig(f'{path}\\{i}.jpg',
                                    bbox_inches = None)
                        
                        self.status_label3.config(text = f"{i+1}/ {self.end_frame}")
                        self.root.update()

                    plt.close(fig)

                    #save the all temperature data as npy file
                    np.save(self.path + '\\' +'Horizontal_temp.npy', temp_horizontal_avg)
                else:
                    tk.messagebox.showerror("Invalid Input","Please enter the width and height of the ROI")
            else:
                tk.messagebox.showerror("Invalid Input","Please enter the coordinates of the ROI")
        else:
            tk.Messagebox.showerror("Invalid Input","Please select the start and end frame")

    #def read_temp_hor(self):

    #Customable ROI area, 

    def show_ver_temp(self):
        self.ver_path = self.path+ '\\'+"Vertical_temp"
        if os.path.exists(self.ver_path):
            # Check if the folder is not empty
            if os.listdir(self.ver_path):
                self.show_ver_enabled = True
                # Set flag to True
                #pump up a new window to show the temperature fig
                ver_temp_window = tk.Toplevel(self.root)
                ver_temp_window.title("Vertical Temperature")
                ver_temp_window.geometry("800x640")
                #read the figure from the folder
                ver_image_path = os.path.join(self.ver_path, f"{self.slider.get()}.jpg")
                # Create a new canvas to display the figure
                self.canvas_ver = tk.Canvas(ver_temp_window, width=600, height=400)
                self.canvas_ver.pack(fill=tk.BOTH, expand=True)
                # Create a new image object 
                image_ver = Image.open(ver_image_path)
                self.current_frame_ver = ImageTk.PhotoImage(image_ver)
                # Display the image on the canvas
                self.canvas_ver.create_image(0, 0, anchor='nw', image=self.current_frame_ver)
                current_value = self.slider.get()

                # Force update of the current image
                self.update_image(current_value)
            else:
                tk.messagebox.showerror("Error","Vertical temperature folder is empty")
        else:
            tk.messagebox.showerror("Error","Vertical temperature folder does not exist")

    def show_hor_temp(self):
        self.hor_path = self.path+ '\\'+"Horizontal_temp"
        if os.path.exists(self.hor_path):
            # Check if the folder is not empty
            if os.listdir(self.hor_path):
                # Set flag to True
                self.show_hor_enabled = True
                #pump up a new window to show the temperature fig
                hor_temp_window = tk.Toplevel(self.root)
                hor_temp_window.title("Horizontal Temperature")
                hor_temp_window.geometry("800x640")
                #read the figure from the folder
                hor_image_path = os.path.join(self.hor_path, f"{self.slider.get()}.jpg")
                # Create a new canvas to display the figure
                self.canvas_hor = tk.Canvas(hor_temp_window, width=600, height=400)
                self.canvas_hor.pack(fill=tk.BOTH, expand=True)
                # Create a new image object
                image_hor = Image.open(hor_image_path)
                self.current_frame_hor = ImageTk.PhotoImage(image_hor)
                # Display the image on the canvas
                self.canvas_hor.create_image(0, 0, anchor='nw', image=self.current_frame_hor)
                current_value = self.slider.get()
                
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
