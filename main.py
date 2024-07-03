import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import uiautomator2 as u2
import subprocess
import time


# Start scrcpy in a subprocess
def launch_scrcpy():
    scrcpy_process = subprocess.Popen(['scrcpy/scrcpy.exe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

    return cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)


# Function to detect specific red regions with white content
def detect_red_regions_with_white(image):
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
        # Apply size constraints
        if 10 < w < 100 and 10 < h < 100:
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


# Function to perform actions on detected regions
def interact_with_regions(device, regions, window_bbox):
    for (x, y, w, h) in regions:
        screen_x = window_bbox[0] + x + w // 2
        screen_y = window_bbox[1] + y + h // 2
        print(f"Clicking on region at ({screen_x}, {screen_y})")
        device.click(screen_x, screen_y)


# Function to plot detected regions
def plot_detected_regions(image, regions):
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Detected Regions', image)


def click(device, x, y):
    device.click(x, y)


def openEatventureFromClearScreen(device):
    # turn on eatventure
    click(device, 950, 1300)
    time.sleep(1.5)
    click(device, 759, 1250)


def main():
    # Launch scrcpy
    scrcpy_process = launch_scrcpy()

    # Give scrcpy some time to start
    time.sleep(3)

    # Connect to the device using uiautomator2
    device = u2.connect()

    # openEatventureFromClearScreen(device)

    # 890 x 2000 -- upgrades coordinates
    # 978 x 2180 -- center upgrades coordinates

    while True:
        try:
            # Capture and crop the scrcpy window
            image = capture_and_crop_scrcpy_window("SM-S908B")

            if image is not None:
                # Detect red regions in the captured screen
                regions = detect_red_regions_with_white(image)

                if regions:
                    print(f"Found {len(regions)} red regions on the screen.")
                    # Plot the detected regions on the image
                    plot_detected_regions(image, regions)
                else:
                    print("No red regions found on the screen.")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to capture and crop scrcpy window.")
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    # Clean up
    cv2.destroyAllWindows()
    scrcpy_process.terminate()


if __name__ == "__main__":
    main()
