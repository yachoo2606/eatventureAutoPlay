import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import subprocess
import time
import re
import keyboard
from multiprocessing import Process, Queue, Event


# Start scrcpy in a subprocess with specified resolution
def launch_scrcpy(max_width, max_height):
    scrcpy_process = subprocess.Popen(['scrcpy/scrcpy.exe', '--max-size', str(max_width), '--max-size', str(max_height)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return scrcpy_process


# Capture the screen and crop to scrcpy window using pyautogui
def capture_and_crop_scrcpy_window(window_title='scrcpy'):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise Exception(f"No window found with title {window_title}")

    scrcpy_window = windows[0]
    bbox = (scrcpy_window.left, scrcpy_window.top, scrcpy_window.right, scrcpy_window.bottom)

    # Capture the screen
    screenshot = pyautogui.screenshot()

    # Convert the screenshot to a numpy array
    screenshot_np = np.array(screenshot)

    # Crop the image to the scrcpy window
    cropped_image = screenshot_np[scrcpy_window.top:scrcpy_window.bottom, scrcpy_window.left:scrcpy_window.right]

    if cropped_image.size == 0:
        raise Exception("Cropped image is empty. Check the window coordinates.")

    return cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR), bbox


# Function to get device resolution using adb
def get_device_resolution():
    result = subprocess.run(['adb/adb.exe', 'shell', 'wm', 'size'], capture_output=True, text=True)
    output = result.stdout.strip()

    # Use regex to find the correct resolution line
    match = re.search(r'Physical size:\s*(\d+x\d+)', output)
    if match:
        size_str = match.group(1)
        width, height = map(int, size_str.split('x'))
        return width, height
    raise Exception("Failed to get device resolution")


# Function to detect specific red regions with white content
def detect_red_regions_with_white(image, exclude_bbox):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range and create a mask
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Apply size constraints and exclude specific region
        if 10 < w < 100 and 10 < h < 100 and not (
                exclude_bbox[0] < x < exclude_bbox[2] and exclude_bbox[1] < y < exclude_bbox[3]):
            # Check for white content within the region
            region = image[y:y + h, x:x + w]
            if contains_white(region):
                detected_regions.append((x, y, w, h))

    return detected_regions


# Function to check for the presence of white content in the region
def contains_white(region):
    # Convert to HSV color space
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Define white color range and create a mask
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Calculate the percentage of white pixels in the region
    white_ratio = cv2.countNonZero(mask) / (region.shape[0] * region.shape[1])
    return white_ratio > 0.2  # Adjust the threshold as needed


# Function to check for the presence of yellow content in the region
def contains_yellow(region):
    # Convert to HSV color space
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Define yellow color range and create a mask
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Calculate the percentage of yellow pixels in the region
    yellow_ratio = cv2.countNonZero(mask) / (region.shape[0] * region.shape[1])
    return yellow_ratio > 0.1  # Adjust the threshold as needed


# Function to detect blue buttons with yellow content
def detect_blue_buttons_with_yellow(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define blue color range and create a mask
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_buttons = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Check for yellow content within the region
        region = image[y:y + h, x:x + w]
        if contains_yellow(region):
            detected_buttons.append((x, y, w, h))

    return detected_buttons


# Function to plot detected regions and exclusion bbox
def plot_detected_regions(image, regions, exclude_bbox, blue_buttons=[]):
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Plot the exclusion bbox
    print(f"Exclusion bbox coordinates: {exclude_bbox}")  # Debug print to verify coordinates
    cv2.rectangle(image, (exclude_bbox[0], exclude_bbox[1]), (exclude_bbox[2], exclude_bbox[3]), (0, 0, 255), 2)
    for (bx, by, bw, bh) in blue_buttons:
        cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
    cv2.imshow('Detected Regions', image)
    cv2.waitKey(1)


# Function to handle the drawing process
def drawing_process(queue, stop_flag):
    while not stop_flag.is_set():
        if not queue.empty():
            image, regions, exclude_bbox, blue_buttons = queue.get()
            plot_detected_regions(image, regions, exclude_bbox, blue_buttons)
        time.sleep(0.1)


def main():
    # Get device resolution
    device_width, device_height = get_device_resolution()
    print(f"Device resolution: {device_width}x{device_height}")

    # Launch scrcpy at the device's resolution
    scrcpy_process = launch_scrcpy(device_width, device_height)

    # Give scrcpy some time to start
    time.sleep(3)

    # Calculate scaling factors
    scrcpy_width, scrcpy_height = device_width, device_height  # Since scrcpy is running at the device's resolution
    scaling_factors = (device_width / scrcpy_width, device_height / scrcpy_height)
    print(f"Scaling factors: {scaling_factors}")

    # Define the exclusion region (adjust these coordinates as needed)
    exclude_bbox = (384, 180, 436, 239)  # Example coordinates for the exclusion region

    # Create queues for inter-process communication
    queue = Queue()
    stop_flag = Event()  # Create a stop flag event

    # Start the drawing process
    drawer = Process(target=drawing_process, args=(queue, stop_flag))
    drawer.start()

    while True:
        try:
            # Check if 'q' key is pressed to stop the program
            if keyboard.is_pressed('q'):
                stop_flag.set()  # Set the stop flag to True to stop the drawing process
                break

            # Capture and crop the scrcpy window
            image, bbox = capture_and_crop_scrcpy_window("SM-S908B")

            if image is not None:
                # Detect red regions in the captured screen
                regions = detect_red_regions_with_white(image, exclude_bbox)

                if regions:
                    print(f"Found {len(regions)} red regions on the screen.")

                    for (x, y, w, h) in regions:
                        screen_x = int((x + w // 2) * scaling_factors[0]) + bbox[0]
                        screen_y = int((y + h // 2) * scaling_factors[1]) + bbox[1]
                        print(f"Clicking on region at ({screen_x}, {screen_y})")
                        pyautogui.click(screen_x, screen_y)
                        time.sleep(0.5)  # Reduced the sleep time

                        # Capture the new window and click blue buttons with yellow content
                        while True:
                            new_image, _ = capture_and_crop_scrcpy_window("SM-S908B")
                            blue_buttons = detect_blue_buttons_with_yellow(new_image)
                            if blue_buttons:
                                print(f"Found {len(blue_buttons)} blue buttons with yellow content on the screen.")
                                # Send the image and detected regions to the drawing process
                                queue.put((new_image, [], exclude_bbox, blue_buttons))

                                for (bx, by, bw, bh) in blue_buttons:
                                    screen_bx = int((bx + bw // 2) * scaling_factors[0]) + bbox[0]
                                    screen_by = int((by + bh // 2) * scaling_factors[1]) + bbox[1]
                                    print(f"Clicking on blue button with yellow content at ({screen_bx}, {screen_by})")
                                    pyautogui.click(screen_bx, screen_by)
                                    time.sleep(0.05)  # Wait for the button to be processed
                            else:
                                break  # No more blue buttons with yellow content to click

                    # Send the image and detected regions to the drawing process
                    queue.put((image, regions, exclude_bbox, []))

                else:
                    print("No red regions found on the screen.")
            else:
                print("Failed to capture and crop scrcpy window.")

        except KeyboardInterrupt:
            stop_flag.set()  # Set the stop flag to True to stop the drawing process
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    # Clean up
    cv2.destroyAllWindows()
    scrcpy_process.terminate()
    drawer.join()  # Ensure the drawing process has finished
    exit(1)


if __name__ == "__main__":
    main()
