from controller import Supervisor,Camera, Display,GPS,Compass ,CameraRecognitionObject  # Importing the Supervisor class from the controller module
import math  # Importing the math module
import numpy as np  # Importing the numpy module and aliasing it as np
from matplotlib import pyplot as plt  # Importing the pyplot submodule from matplotlib and aliasing it as plt
from scipy import signal  # Importing the signal submodule from scipy

# Define constants
WP = [(0.61,0),(0.61,-2.32),(0.35,-3.2),(-1.71,-3.2),(-1.71,0.24),(-1.33,0.69),(-0.78,0.48),(-1.09,0.29),(-1.7,-0.49),(-1.7,-3.22),(0.66,-3.22),(0.66,0),(0,0)]
# Waypoints
index = 0  # Index variable for waypoints
robot = Supervisor()  # Create a Supervisor instance

timestep = int(robot.getBasicTimeStep())  # Get the timestep

# Get motor devices and set initial configuration
left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Get LIDAR device and enable it
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(timestep)
lidar.enablePointCloud()
#camera 
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)
num_objects = camera.getRecognitionNumberOfObjects()
camera.enableRecognitionSegmentation()
display_1 = robot.getDevice('display(1)')
width = camera.getWidth()
height = camera.getHeight()
# Get display device
display = robot.getDevice("display")
display.setColor(0x0000FF)  # Set color
display.drawPixel(150, 150)  # Draw a pixel at coordinates (150, 150)

# Define kernel values for convolution
kernel_values = [1, 30]

# Initialize variables for velocity and position
vl, vr = 0, 0
dx = 0
w = 0
pos = [0, 0, 0]
xw = 0
yw = 0.0277
dw = 0
theta = 1.57
world = 6
angles = np.linspace(3.1415/1.5, -3.1415/1.5, 667)
x_r, y_r = [], []
x_w, y_w = [], []

# Get GPS device and enable it
gps = robot.getDevice('gps')
gps.enable(timestep)

# Get size of the display
size_x = display.getWidth()
size_y = display.getHeight()

# Get compass device and enable it
compass = robot.getDevice('compass')
compass.enable(timestep)

# Function to convert world coordinates to map coordinates
def world2map(xw, yw, worldSize=1.0, sx=300, sy=300):
    if not np.isfinite(xw) or not np.isfinite(yw):
        return [0, 0]  # Return [0, 0] if coordinates are not finite

    mapSize_x = sx
    mapSize_y = sy
    center = [-0.5, -1.5]
    normalizedX = (-center[0] + xw) / worldSize
    normalizedY = (center[1] - yw) / worldSize
    px = 150 + int(normalizedX * (mapSize_x - 1))
    py = 150 + int(normalizedY * (mapSize_y - 1))
    px = min(max(px, 0), mapSize_x - 1)
    py = min(max(py, 0), mapSize_y - 1)

    return [px, py]

# Initialize map
map = np.zeros((300, 300))

# Set marker position
marker = robot.getFromDef("Marker").getField("translation")
marker.setSFVec3f([0, 0, 0.2])

finish = 0  # Flag to indicate if the mission is finished

# Main loop
while robot.step(timestep) != -1:
    #camera data 
    data = camera.getImage()
    objects = camera.getRecognitionObjects()
    #print(CameraRecognitionObject.getId())
    number_of_objects = camera.getRecognitionNumberOfObjects()
    print(f'Recognized {number_of_objects} objects.')
    #camera = getPosition()
    counter = 1
    for object in objects:
      
      id_object = object.getId()
      position = object.getPosition()
      position = object.getPosition()
      orientation = object.getOrientation()
      size = object.getSize()
      position_on_image = object.getPositionOnImage()
      size_on_image = object.getSizeOnImage()
      number_of_colors = object.getNumberOfColors()
      colors = object.getColors()
      
      
      #print(f' Object {counter}/{number_of_objects}: {object.getModel()} (id = {object.getId()})')
      #print(f' Position: {position[0]} {position[1]} {position[2]}')
      #print(f' Orientation: {object.orientation[0]} {orientation[1]} {orientation[2]} {orientation[3]}')
      #print(f' Size: {size[0]} x {size[1]}')
      #print(f' Position on camera image: {position_on_image[0]} {position_on_image[1]}')
      #print(f' Size on camera image: {size_on_image[0]} x {size_on_image[1]}')
      #print(object.getPositionOnImage())
      #print(id_object)
      
      #print ('id_object ' + str(id_object))
    
    if data:
        ir = display_1.imageNew(data, Display.BGRA, width, height)
        display_1.imagePaste(ir, 0, 0, False)
        display_1.imageDelete(ir)




    # Get GPS position and compass heading
    xw = gps.getValues()[0]
    yw = gps.getValues()[1]
    theta = np.arctan2(compass.getValues()[0], compass.getValues()[1])

    # Set marker position to current waypoint
    marker.setSFVec3f([*WP[index], 0])

    # Calculate distance and angle to current waypoint
    rho = np.sqrt(((xw - WP[index][0]) ** 2 + (yw - WP[index][1]) ** 2))
    alpha = np.arctan2(WP[index][1] - yw, WP[index][0] - xw) - theta

    # Adjust angle if it exceeds pi
    if alpha > np.pi:
        alpha = alpha - 2 * np.pi

    # Check if the robot reached the waypoint
    if rho < 0.3:
        index += 1
        if index > len(WP) - 1:
            index = 0
            finish = 1

    # Print distance and angle to waypoint
    print(rho, ",", alpha / 3.1416 * 180)

    # Get LIDAR range image
    ranges = lidar.getRangeImage()
    ranges[ranges == np.inf] = 100

    # Transform LIDAR data to world coordinates
    x_r, y_r = [], []
    x_w, y_w = [], []
    w_T_r = np.array([[np.cos(theta), -np.sin(theta), xw],
                      [np.sin(theta), np.cos(theta), yw],
                      [0, 0, 1]])
    X_i = np.array([ranges * np.cos(angles) + 0.202, ranges * np.sin(angles), np.ones(667,)])
    D = w_T_r @ X_i
 
    px_marker, py_marker = world2map(WP[index][0], WP[index][1], world, size_x, size_y)
    display.setColor(0x00ff00)
    #display.drawPixel(px_marker, py_marker)
    display.fillRectangle( px_marker, py_marker,px_marker/20,py_marker/30)
    
  
    # Map using probabilistic method
    for i in range(len(D[0])):
        if D[0, i] != xw and D[1, i] != yw and i > 80 and i < 587:
            px, py = world2map(D[0, i], D[1, i], world, size_x, size_y)
            map[px, py] += 0.1
            v = int(map[px, py] * 255)
            if v >= 255:
                v = 254
            color = 0x010101 * v
            display.setColor(color)
            display.drawPixel(px, py)

    # Draw current position on map
    px, py = world2map(xw, yw, world, size_x, size_y)
    display.setColor(0xFF0000)
    display.drawPixel(px, py)

    # Controller - Proportional-Derivative Control
    p1 = 6
    p2 = 2
    vl = (-alpha * p1 + rho * p2)
    vr = (alpha * p1 + rho * p2)

    # Limit velocities
    if vl > 10.1523:
        vl = 10.1523
    if vr > 10.1523:
        vr = 10.1523

    # Set motor velocities
    left_motor.setVelocity(vl)
    right_motor.setVelocity(vr)

    # Check if mission is finished
    if finish == 1:
        # Plot maps with different kernel sizes
        fig, axes = plt.subplots(1, len(kernel_values), figsize=(15, 5))
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        for i, kernel_value in enumerate(kernel_values):
            kernel = np.ones((kernel_value, kernel_value))
            cmap = signal.convolve2d(map, kernel, mode='same')
            cspace = cmap > 0.9
            
            # Mirror image
            cspace_mirror = np.fliplr(cspace)
            cspace_mirror = np.rot90(cspace_mirror)
            axes[i].imshow(cspace_mirror)
            axes[i].set_title(f"Kernel {kernel_value}")

        plt.tight_layout()
        plt.show()