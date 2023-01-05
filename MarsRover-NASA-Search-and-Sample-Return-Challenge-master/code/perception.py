import numpy as np
import cv2
import time

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def navigable_thresh(img, rgb_thresh=(160, 160, 160)):
    
    # Create an array of zeros with the same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    
    above_thresh = ((img[:,:,0] > rgb_thresh[0]) &
                    (img[:,:,1] > rgb_thresh[1]) &
                    (img[:,:,2] > rgb_thresh[2]))
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    return color_select


def obstacle_thresh(img, rgb_thresh=(160, 160, 160)):
   
    # Create an array of zeros with the same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be below all three threshold values in rbg_thresh.
    #   Values below the threshold will now contain a boolean array with TRUE.
    below_thresh = ((img[:,:,0] < rgb_thresh[0]) &
                    (img[:,:,1] < rgb_thresh[1]) &
                    (img[:,:,2] < rgb_thresh[2]))
    # Index the array of zeros with the boolean array and set to 1
    color_select[below_thresh] = 1
    return color_select


def rock_thresh(img):
   
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV, 3)
    
    # Define range of yellow colors in HSV
    lower_yellow = np.array([20, 150, 100], dtype='uint8')
    upper_yellow = np.array([50, 255, 255], dtype='uint8')
    
    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask 


def rover_coords(binary_img):
   
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Translate the pixel positions with reference to the rover position being
    #   at the center bottom of the image. Must flip the xpos and ypos.
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


def to_polar_coords(x_pixel, y_pixel):
    
    # Calculate distance to each pixel using pythagoream theorem
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel using arc tangent
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


def rotate_pix(xpix, ypix, yaw):

    # Convert yaw degrees to radians
    yaw_rad = yaw * np.pi / 180
    # Apply a matrix rotation counter-clockwise by the yaw radians
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    
    # Scale and translate the rotated pixel arrays
    xpix_translated = (xpos + (xpix_rot / scale))
    ypix_translated = (ypos + (ypix_rot / scale))
    return xpix_translated, ypix_translated


def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale=10):
    
    # Rotate the coordinates so x and y axes are parallel to the world axes
    xpix_rot, ypix_rot = rotate_pix(xpix=xpix, ypix=ypix, yaw=yaw)
    # Translate the rotated arrays by the rover's location in the world
    xpix_tran, ypix_tran = translate_pix(xpix_rot=xpix_rot, ypix_rot=ypix_rot,
                                         xpos=xpos, ypos=ypos, scale=scale)
    # Clip the array ranges to fit within the world map
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    return x_pix_world, y_pix_world


def perspect_transform(img, src, dst):
 
    M = cv2.getPerspectiveTransform(src, dst)
    # Keep same size as input image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return warped


def perception_step(Rover):
   
    # Camera image from the current Rover state (Rover.img)
    img = Rover.img
   
   
    dst_size = 5 
   
    bottom_offset = 6 # was 6
    src = np.float32([[13, 140], [302, 140], [200, 96], [118, 96]])
    dst = np.float32([
        [img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
        [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
        [img.shape[1]/2 + dst_size, img.shape[0] - 2 * dst_size - bottom_offset],
        [img.shape[1]/2 - dst_size, img.shape[0] - 2 * dst_size - bottom_offset]])

   
    warped = perspect_transform(img=img, src=src, dst=dst)

   
    navigable = navigable_thresh(img=warped, rgb_thresh=(160, 160, 160)) 
    obstacles = obstacle_thresh(img=warped, rgb_thresh=(160, 160, 160)) 
    rock_samples = rock_thresh(img=warped)
     #ignor bad data 
    navigable[0:int(navigable.shape[0]/2),:] = 0
    obstacles[0:int(obstacles.shape[0]/2),:] = 0
    
    Rover.vision_image[:,:,0] = obstacles * 255
    Rover.vision_image[:,:,1] = rock_samples * 255
    Rover.vision_image[:,:,2] = navigable * 255

   
    navigable_xpix, navigable_ypix = rover_coords(binary_img=navigable)
    obstacles_xpix, obstacles_ypix = rover_coords(binary_img=obstacles)
    rocks_xpix, rocks_ypix = rover_coords(binary_img=rock_samples)

    
    scale = 20#  was 2*dist_size
    xpos, ypos = Rover.pos
    yaw = Rover.yaw
    worldmap_size = Rover.worldmap.shape[0]

    navigable_x_world, navigable_y_world = pix_to_world(
        xpix=navigable_xpix, ypix=navigable_ypix,
        xpos=xpos, ypos=ypos, yaw=yaw, world_size=worldmap_size, scale=scale)
    obstacles_x_world, obstacles_y_world = pix_to_world(
        xpix=obstacles_xpix, ypix=obstacles_ypix,
        xpos=xpos, ypos=ypos, yaw=yaw, world_size=worldmap_size, scale=scale)
    rocks_x_world, rocks_y_world = pix_to_world(
        xpix=rocks_xpix, ypix=rocks_ypix,
        xpos=xpos, ypos=ypos, yaw=yaw, world_size=worldmap_size, scale=scale)

    
    if (Rover.pitch < 2 or Rover.pitch > 358) and (Rover.roll < 1 or Rover.roll > 359):
        # Limit world map updates to only images that have limited roll and pitch
        Rover.worldmap[obstacles_y_world, obstacles_x_world, 0] += 1
        Rover.worldmap[rocks_y_world, rocks_x_world, 1] = 255
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    
    distances, angles = to_polar_coords(x_pixel=navigable_xpix,
                                        y_pixel=navigable_ypix)
    
    # Update Rover pixel distances and angles
    Rover.nav_dists = distances
    Rover.nav_angles = angles 
    
    if len(rocks_xpix) > 5:# size width of rock
        # If a rock is identified, make the rover navigate to it
        rock_distance, rock_angle = to_polar_coords(x_pixel=rocks_xpix,
                                                    y_pixel=rocks_ypix)
        Rover.rock_dist = rock_distance
        Rover.rock_angle = rock_angle 
        if not Rover.sample_seen:
            # First frame sample has been seen, thus start the sample timer
            Rover.sample_timer = time.time()
        Rover.sample_seen = True

    if Rover.start_pos is None:
        Rover.start_pos = (Rover.pos[0], Rover.pos[1])
        print('STARTING POSITION IS: ', Rover.start_pos)
        
    image_list = [img,warped, navigable, obstacles, rock_samples]
    cv2.imshow("warped",warped)
    cv2.imshow("navigable",navigable*255)
    cv2.imshow("rock",rock_samples)
    cv2.imshow("obstacle",obstacles*255)
    cv2.imshow("image",img)
    cv2.waitKey(5)

    
    return Rover
