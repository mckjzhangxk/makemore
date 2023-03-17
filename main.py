# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    cp=cv2.VideoCapture('/Library/Frameworks/GStreamer.framework/Commands/gst-launch-1.0 videotestsrc ! appsink',cv2.CAP_GSTREAMER)

    while True:
        ok,I=cp.read()
        cv2.imshow("a",I)
        key=cv2.waitKey(2)

        if key==ord('q'):
            break
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
