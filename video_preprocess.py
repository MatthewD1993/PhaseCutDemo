# from tempfile import tempdir
# from webbrowser import get

from atexit import register

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



ob_box = [[],[]]
def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
      print ('Start Mouse Position: '+str(x)+', '+str(y))
      sbox = [x, y]
      ob_box[0] = sbox

    elif event == cv2.EVENT_LBUTTONUP:
      print ('End Mouse Position: '+str(x)+', '+str(y))
      ebox = [x, y]
      ob_box[1] = ebox
      cv2.rectangle(temp, ob_box[0], ob_box[1], (0,255,0),2, 8)
      cv2.imshow("Window", temp)



# For webcam input:
cap = cv2.VideoCapture("/mnt/sdcard/IMG_9995.MOV")
register(cap.release)



ret, ori_im = cap.read()
image_height, image_width = ori_im.shape[:2]

out_file = './res.mp4'
out_cap = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640, 360))
register(out_cap.release)

# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# cap.set(3,1280)
# cap.set(4,720)
temp = ori_im.copy()
cv2.namedWindow('Window')
cv2.setMouseCallback('Window', on_mouse, 0)

k=0
temp = ori_im.copy()
while k != 'y':
  # Display the image
  cv2.imshow("Window", temp)
  k = chr(cv2.waitKey(0))
  # If r is pressed, clear the window, using the dummy image
  if (k == 'r'):
    temp = ori_im.copy()
    cv2.imshow("Window", temp)

cv2.destroyAllWindows()
ob_area = ori_im[ob_box[0][1]:ob_box[1][1],ob_box[0][0]:ob_box[1][0]]
cv2.imshow('ob_area', ob_area)
cv2.waitKey(0)


from hsv_trackbar import get_hsv_seg_func
hsv_seg_func = get_hsv_seg_func(ob_area)
empty_mask, res = hsv_seg_func(ob_area)


#### Loop ####
from helper import item_in_box_wrapper, hands_in_box_area_func
item_in_box_func = item_in_box_wrapper(hsv_seg_func, empty_mask)

# cv.putText(	img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]	) ->	img
pos_args = [(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2., (255, 0, 0),  2, cv2.LINE_AA]
signal_hand = []
signal_book = []
def save_signals():
  import pickle as pkl
  res = {'signal_hand': signal_hand, 'signal_book': signal_book}
  with open('signal.pkl', 'wb') as f:
    pkl.dump(res, f)
  return
register(save_signals)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    ) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.

    # cv2.flip(image, 1),
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    is_book_in_box_area = item_in_box_func(image[ob_box[0][1]:ob_box[1][1],ob_box[0][0]:ob_box[1][0]])
    is_hand_in_box_area = False
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        is_hand_in_box_area = hands_in_box_area_func(hand_landmarks, image_width, image_height, ob_box[0], ob_box[1])
        if is_hand_in_box_area:
          break
    signal_hand.append(is_hand_in_box_area)
    signal_book.append(is_book_in_box_area)
    text = f''' Book IN: {"yes" if is_book_in_box_area else "no"}   Hands IN: {"yes" if is_hand_in_box_area else "no"}'''
    text = f''' Book IN: {is_book_in_box_area}   Hands IN: {"yes" if is_hand_in_box_area else "no"}'''
    cv2.rectangle(image, ob_box[0], ob_box[1], (0,255,0),2, 8)
    image = cv2.putText(image, text, *pos_args)
    
    image = cv2.resize(image, (640, 360))
    cv2.imshow('MediaPipe Hands', image)
    out_cap.write(image)
    # if cv2.waitKey(1) & 0xFF == 27: 
    # Press key: ESC
    if cv2.waitKey(1) == 27: 
      break