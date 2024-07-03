import cv2

# Load the image
image_path = 'test.png'
image = cv2.imread(image_path)


# Display the image and wait for a key press to get the coordinates
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked coordinates: x={x}, y={y}")


cv2.imshow('image', image)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
