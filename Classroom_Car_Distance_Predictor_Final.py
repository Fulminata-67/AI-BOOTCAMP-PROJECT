import cv2 as cv
import numpy as np
import pandas as pandas
import os
from skimage.transform import resize
from skimage.io import imread
from matplotlib import pyplot as plt
from sklearn import svm 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.svm import SVC 

from numpy import asarray
import mailcap

from PIL import Image
from PIL import ImageDraw


#The basic idea of the model is to take a picture every 3 seconds and interpret if there is 
#enough space to overtake
#The photo-taking model isn't yet implemented as it could not be tested


categories = ["Road", "Cars"]
inputArray = []
outputArray = []
datadir1 = r"C:\Users\Computerlab\Desktop\AI-BOOTCAMP-PROJECT-main\ImagesForProject" #path of the images
datadir = r"AI BOOTCAMP PROJECT/ImagesForProject"

for i in categories:
    print(f"Loading....Category:{i}")
    path = os.path.join(datadir, i)
    for image in os.listdir(path):
        image_array = imread(os.path.join(path,image))
        image_resized = resize(image_array, (150, 150, 3))
        inputArray.append(image_resized.flatten())
        outputArray.append(categories.index(i))
    print(f"Loaded Category:{i} Successfully")
    flatData = np.array(inputArray)
    target=np.array(outputArray)

#dataframe
df = pandas.DataFrame(flatData)
df["Target"] = target
df.shape

print(df)

x = df.iloc[:, :-1] #input data
y = df.iloc[:, -1] #output data

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=77, stratify=y)

model = SVC()
model.fit(x_train, y_train)

#print results
predictions = model.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
print(classification_report(y_test, predictions))
print("The percentage of this model is (in %):", accuracy*100)

#Testing what happens when a car is infront of the truck
path = "AI BOOTCAMP PROJECT/ImagesForProject/Cars/car_04.png"
image = imread(path)
image_resized = resize(image, (150, 150, 3))
l = [image_resized.flatten()]
result = categories[model.predict(l)[0]]
print(result)

path1 = cv.imread(path)
cv.imshow("What the truck driver sees", path1)

train = "AI BOOTCAMP PROJECT/ImagesForProject/Car2_test.jpeg"
truck = "AI BOOTCAMP PROJECT/ImagesForProject/Truck_img.jpg"


#DISPLAY

if result == "Road":
    truck1 = cv.imread(truck)
    truck2 = cv.circle(truck1, (270, 120), 5, (0, 255, 0), 5)
    cv.imshow("What the driver of the vehicle behind sees", truck2)
    cv.waitKey(0)
    cv.destroyAllWindows()

if result == "Cars":
    query = cv.imread(path)
    train = cv.imread("AI BOOTCAMP PROJECT/ImagesForProject/Car2_test.jpeg")
    truck1 = cv.imread(truck)
    truck2 = cv.circle(truck1, (270, 120), 5, (0, 0, 255), 5)
    cv.imshow("What the driver of the vehicle behind sees", truck2)
    cv.waitKey(0)
    cv.destroyAllWindows()