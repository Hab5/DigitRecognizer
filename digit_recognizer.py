import matplotlib.pyplot as plt
from tensorflow import keras
from tkinter import *
import numpy as np
import cv2
import os

model = keras.models.load_model('./cnn_model.h5')

class DigitRecognizer(object):

    DEFAULT_PEN_SIZE = 15.0
    DEFAULT_COLOR = 'white'

    def __init__(self):
        self.root = Tk()
        self.root.title("Digit Recognizer")
        self.buttons_labels_canvas()
        self.setup()
        self.root.lift()
        self.root.attributes('-topmost',True)
        self.root.after_idle(self.root.attributes,'-topmost',False)
        self.root.mainloop()

    def buttons_labels_canvas(self):
        self.guess = Button(self.root, text='GUESS', command=self.guess,
                                                font=('verdana 10 bold'))
        self.clear = Button(self.root, text='CLEAR', command=self.clear,
                                                font=('verdana 10 bold'))
        self.text = Label(self.root, text="WRITE A DIGIT BETWEEN 0 AND 9",
                                                font=('verdana 10 bold'))
        self.c = Canvas(self.root, bg='black', width=400, height=400)
        self.c.grid(row=1, columnspan=10)
        self.guess.grid(row=0, column=0)
        self.clear.grid(row=0, column=9)
        self.text.grid(row=0, column=4)

    def setup(self):
        self.run = 1
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def guess(self):
        if self.run == 1:
            self.run = 0
            self.get_image()
            self.process_image()
            self.display_mnist()
            self.make_prediction()

    def get_image(self): # Canvas to postscript then .png
        self.c.postscript(file="/tmp/mnist.ps", colormode='color')
        os.system('convert /tmp/mnist.ps +profile "icc" /tmp/mnist.png')

    def display_mnist(self): # Get the png, apply color map and print to canvas
        from PIL import ImageTk, Image
        im = Image.fromarray(self.mnist_user_image*255).convert('RGB')
        im.save("/tmp/mnist.png")
        os.system('convert /tmp/mnist.png -resize 28x28 /tmp/mnist.png')
        self.mnist_user_image = np.array(plt.imread('/tmp/mnist.png'),
                                                    dtype=np.float64)
        cm = plt.get_cmap('hot')
        self.cmap_mnist = cm(self.mnist_user_image)
        self.image = Image.fromarray((self.cmap_mnist[:,:,:3]*255).astype(np.uint8))
        self.image = self.image.resize((408, 408))
        self.image = ImageTk.PhotoImage(self.image)
        self.c.create_image(0, 0, anchor=NW, image=self.image)

    def make_prediction(self): # Call the trained keras model and predict
        self.mnist_user_image = self.mnist_user_image.reshape((1,28,28,1))
        prediction = model.predict_classes(self.mnist_user_image)
        self.text.configure(text=f'MODEL PREDICTION: {prediction}')

    def process_image(self):
        self.mnist_user_image = np.array(plt.imread('/tmp/mnist.png'),
                                              dtype=np.float64)[:,:,3]
        self.crop()
        self.pad_resize()

    def crop(self): # Get the cropbox of the digit
        non_empty_columns = np.where(self.mnist_user_image.max(axis=0)>0)[0]
        non_empty_rows = np.where(self.mnist_user_image.max(axis=1)>0)[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows),
                   min(non_empty_columns), max(non_empty_columns))
        self.mnist_user_image = self.mnist_user_image[cropBox[0]:cropBox[1]+1,
                                                      cropBox[2]:cropBox[3]+1]

    def pad_resize(self): # Cropbox to 280x280 with the same aspect ratio, add border
        desired_size = 280
        old_size = (self.mnist_user_image.shape[:2])
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        self.mnist_user_image = cv2.resize(self.mnist_user_image,
        dsize=(new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC)
        delta_w = desired_size + 120 - new_size[1]
        delta_h = desired_size + 120 - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        self.mnist_user_image = cv2.copyMakeBorder(self.mnist_user_image,
            top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

    def clear(self):
        self.text.configure(text=f'WRITE A DIGIT BETWEEN 0 AND 9')
        self.c.delete("all")
        self.run = 1

    def paint(self, event):
        if self.old_x and self.old_y and self.run == 1:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                    width=self.DEFAULT_PEN_SIZE, fill=self.DEFAULT_COLOR,
                    capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    DigitRecognizer()
