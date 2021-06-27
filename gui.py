from tensorflow.keras.models import load_model
from tkinter import *
import tkinter as tk
import cv2
import numpy as np
from PIL import ImageGrab
from tensorflow.keras.models import load_model

model = load_model('mnist.h5')

source = Tk()
source.resizable(0, 0)
source.title("Handwritten Digit Recognition")
initx, inity = None, None
image_number = 0


def clear_source():
    global draw_area
    draw_area.delete("all")


def activate_event(event):
    global initx, inity
    draw_area.bind('<B1-Motion>', draw_lines)
    initx, inity = event.x, event.y


def draw_lines(event):
    global initx, inity
    x, y = event.x, event.y
    draw_area.create_line((initx, inity, x, y), width=7, fill='black', capstyle=ROUND, smooth=True, splinesteps=12)
    initx, inity = x, y


def recognize_digit():
    global image_number
    filename = f'image_{image_number}.png'
    widget = draw_area
    x = source.winfo_rootx() + widget.winfo_x()
    y = source.winfo_rooty() + widget.winfo_y()
    x1 = x+widget.winfo_width()
    y1 = y+widget.winfo_height()
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    digit = cv2.imread(filename, cv2.IMREAD_COLOR)
    make_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(
        make_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for ct in contours:
        x, y, w, h = cv2.boundingRect(ct)
        cv2.rectangle(digit, (x, y), (x+w, y+h), (255, 0, 0), 1)
        top = int(0.05*th.shape[0])
        bottom = top
        left = int(0.05*th.shape[1])
        right = left
        roi = th[y-top:y+h+bottom, x-left:x+w+right]
        try:
            img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            print(img.shape)
        except:
            break

        img = img.reshape(1, 28, 28, 1)
        img = img/255.0
        prediction = model.predict([img])[0]
        final = np.argmax(prediction)
        data = str(final)+'  '+str(int(max(prediction)*100))+'%'
        cv2.putText(digit, data, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow('digit', digit)
    cv2.waitKey(0)

draw_area = Canvas(source, width=640, height=480, bg='white')
draw_area.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)
draw_area.bind('<Button-1>', activate_event)
btn_save = Button(text="Recognize the Digit", fg='black', command=recognize_digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text="Clear Area", fg='black', command=clear_source)
button_clear.grid(row=2, column=1, pady=1, padx=1)

source.mainloop()
