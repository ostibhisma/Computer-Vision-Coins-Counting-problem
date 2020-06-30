# importing library for the problem
import cv2
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


class CoinRecognizer:
    """
    This class reads the video frame by frame and process each frame to display the total
    amount of coin in the given video.
    
    """
    
    # initially capturing the video
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        
        
    # getting frame from the video and converted to RGB color channel
    def get_frame(self):
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
        return self.frame
        
    # converted RGB channel to GRAY image and used erosion operation to
    #  get rid of small noisy holes present in frame and masking the frame.
    def get_coins_mask(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel, iterations=5)
        ret, self.coins_mask = cv2.threshold(img, 60, 255, 0)
        return self.coins_mask
        
    # finding the contours of masked frame
    def get_contours(self,coins_mask):
        self.contours, _ = cv2.findContours(coins_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return self.contours
        
    # preparing the data in the list
    def get_data(self,contours,frame):
        self.radii = []
        self.center = []
        for cnt in contours:
            (x,y), radius = cv2.minEnclosingCircle(cnt)
            self.radii.append(round(radius, 6))
            self.center.append((x,y))
            
        self.means_hue = []
        self.means_saturation = []
        self.means_value = []
        
        hsv_image = None
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        for cnt in self.contours:
            mask = np.zeros_like(self.coins_mask)
            m = cv2.drawContours(mask, [cnt], -1, (255), -1)
            (mean_hue, mean_saturation, mean_value, _) = cv2.mean(hsv_image, mask=m)
            self.means_hue.append(mean_hue)
            self.means_saturation.append(mean_saturation)
            self.means_value.append(mean_value)
        return self.radii, self.means_hue, self.means_saturation, self.means_value,self.center
        
    # loading the model
    # model is a pickle file
    # model = Support Vector Machine with C = 0.4 where C is the parameter for the soft margin cost function, which controls the influence of each individual support vector
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        return self.model
        
    # load the data in pandas's dataframe and using the loaded model , predicting the values and
    # appended in the test_labels
    def predict(self):
        self.test_labels = []
        for i in range(len(self.contours)):
            data = {'radius': self.radii[i], 'mean_hue': self.means_hue[i],
                    'mean_saturations': self.means_saturation[i], 'mean_value': self.means_value[i]}
            test_data = pd.DataFrame([data])
            pred = self.model.predict(test_data)
            self.test_labels.append(pred)
        return self.test_labels
        
    # calculating the total amount and shown in the live screen 
    def show(self,frame):
        total_money = np.sum(self.test_labels)
        for (x,y),i in zip(self.center,self.test_labels):
            text1 = f"{str(i)} rupees"
            j = cv2.drawContours(frame, self.contours, -1, (0, 0, 255), 2)
            output = cv2.putText(j, text1,(int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5,cv2.LINE_AA)
        text = f"Total amount is : {total_money}"
        output = cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Output", output)
        
        
    def start(self):
        cap = cv2.VideoCapture("coin.mp4")
        while True:
            _, frame = cap.read()
            if _ == False:
                break
            coins_mask = self.get_coins_mask(frame)
            contours = self.get_contours(coins_mask=coins_mask)
            self.get_data(contours,frame)
            self.predict()
            self.show(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        
if __name__ == '__main__':
    coin_recognizer = CoinRecognizer(model_path="model.pkl")
    coin_recognizer.start()
