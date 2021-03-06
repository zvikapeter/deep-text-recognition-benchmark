import cv2
import numpy as np

cropping = False
FinishROI = 0
roi = []
coordinates = []
x_start, y_start, x_end, y_end = 0, 0, 0, 0

image = None
oriImage = None


def mouse_crop(image_np):
    def mouse_crop_aux(event, x, y, flags, param):
        # grab references to the global variables
        global x_start, y_start, x_end, y_end, cropping, roi, FinishROI, coordinates

        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x_end, y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            x_end, y_end = x, y
            cropping = False  # cropping is finished

            refPoint = [(x_start, y_start), (x_end, y_end)]

            if len(refPoint) == 2:  # when two points were found
                roi.append(oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]])
                coordinates.append([x_start, y_start, x_end, y_end])
                #cv2.imshow("Cropped", roi[-1])
                FinishROI += 1



    image =image_np
    oriImage = image.copy()
    cv2.namedWindow("image")
    cv2.imshow("image", image)

    while FinishROI!=5:
        cv2.setMouseCallback("image", mouse_crop_aux)

        i = image.copy()

        if not cropping:
            cv2.imshow("image", image)

        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)

        cv2.waitKey(1)

    # close all open windows
    cv2.destroyAllWindows()

    roi_resize=[]
    for single_roi in roi:
        roi_resize.append(cv2.resize(single_roi, ( 100, 32)))

    return roi_resize, coordinates


if __name__ == '__main__':
    image = cv2.imread('demo_image/Screenshot_20200322-140545_WhatsApp.jpg')
    oriImage = image.copy()
    R =mouse_crop(image)
    a=1
    # while True:
    #     i = image.copy()
    #
    #     if not cropping:
    #         cv2.imshow("image", image)
    #
    #     elif cropping:
    #         cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
    #         cv2.imshow("image", i)
    #         # return i;
    #
    #     cv2.waitKey(1)
    #
    # # close all open windows
    # cv2.destroyAllWindows()
