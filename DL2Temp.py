import numpy as np
import os
import matplotlib.pyplot as plt
import struct
import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor

path = 'C:\\Users\\Administrator\\Desktop\\Project1\\record\\Peng\\test_1\\'

#Check if the path is empty
if not os.path.exists(path):
    print('The path does not exist')
    exit()
else:
    #read all .bin files in the path and sort
    file_list = sorted(glob.glob(os.path.join(path, "*.bin")))

#Define the frame size
width = 382
height = 288
frame_size = width * height

#total number of frames
total_frames = len(file_list)

#create empty array to store the temperature data
Temp_list = np.zeros((height, width, total_frames))

#dealing with the intial frame, open the binary file
with open(file_list[0], 'rb') as file:
    # Read timestamp (int64)
    initial_timestamp = struct.unpack('q', file.read(8))[0]
    #print('The initial timestamp is:', initial_timestamp)
    # Read temperature data (int16)
    initial_temp = np.frombuffer(file.read(frame_size * 2), dtype=np.int16)/100
    max_temp = np.max(initial_temp)
    min_temp = np.min(initial_temp)

'''
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    for i in tqdm.tqdm(range(1000)):
        with open(file_list[i], 'rb') as file:
            # Read timestamp (int64)
            timestamp = struct.unpack('q', file.read(8))[0]
            #print('The timestamp is:', timestamp)
            # Read temperature data (int16)
            DL_data = np.frombuffer(file.read(frame_size * 2), dtype=np.int16)/100
            #DL_data = np.fromfile(file, dtype=np.int16, count=frame_size).reshape(height, width)/100
            Temp_list[:, :, i] = DL_data.reshape((height, width))
            
            max_temp = np.max(Temp_list)
            min_temp = np.min(Temp_list)
    #max_temp = np.max(Temp_data)
    #min_temp = np.min(Temp_data)

#max_temp = np.max(Temp_data)
#min_temp = np.min(Temp_data)
#save the temperature data
np.save('Temperature_data', Temp_list)

#check if the data is saved
if os.path.exists('Temperature_data.npy'):
    print('The temperature data is converted and saved!')
'''

print('The maximum temperature is:', max_temp)
print('The minimum temperature is:',min_temp)
plt.imshow(initial_temp.reshape((height, width)), cmap='jet', vmin=min_temp, vmax=max_temp)
plt.colorbar()
plt.show()



