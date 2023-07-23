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
datadir = "AI BOOTCAMP PROJECT/ImagesForProject/" #path of the images

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

    query = cv.cvtColor(query, cv.COLOR_BGR2GRAY)
    train2 = cv.cvtColor(train, cv.COLOR_BGR2GRAY)

    SiftModel = cv.xfeatures2d.SIFT_create()

    query_KeyPoints, query_Descriptions = SiftModel.detectAndCompute(query, None)
    train_KeyPoints, train_Descriptions = SiftModel.detectAndCompute(train, None)

    bruteForce = cv.BFMatcher(cv.NORM_L1, crossCheck = True)

    matches = bruteForce.match(train_Descriptions, query_Descriptions)

    x = list()
    y = list()

    for t in matches:
        x1, y1 = train_KeyPoints[t.trainIdx].pt
        x.append(x1)
        y.append(y1)

    x.sort()
    y.sort()

    xStart = int(x[0])
    xEnd = int(x[-1])
    yStart = int(y[0])
    yEnd = int(x[-1])
    yEnd = yEnd
    print(yEnd)

    cropped = query[xStart:xEnd, yStart:yEnd]

    numpydata = asarray(cropped)

    knownWidth = 720
    focalLenth = 175
    print(numpydata.shape)
    y = int(numpydata.shape[1])
    y = y + 100
    print(y)

    distance = (knownWidth * focalLenth)
    distance = distance / y
    distance = distance / 12
    distance = distance / 39.37
    print("The distance of the car from the vehicle is: ", distance, "meters")

    if distance < 4.5: 

        truck1 = cv.imread(truck)
        truck2 = cv.circle(truck1, (270, 120), 5, (0, 0, 255), 5)
        cv.imshow("What the driver of the vehicle behind sees", truck2)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        truck1 = cv.imread(truck)
        truck2 = cv.circle(truck1, (270, 120), 5, (0, 255, 0), 5)
        cv.imshow("What the driver of the vehicle behind sees", truck2)
        cv.waitKey(0)
        cv.destroyAllWindows()

#THE BELOW CODE IS FOR REFERENCE FOR FUTURE PROCEEDINGS
# initialize the camera
#cap = cv2.VideoCapture(0)

# set the frame size
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# main loop
#while True:
    # wait for 3 seconds
    #time.sleep(3)

    # capture a frame from the camera
    #ret, frame = cap.read()

    # check if the frame was successfully captured
    #if ret:
        # save the image
        #cv2.imwrite("image.jpg", frame)

    # exit the program if the user presses the 'q' key
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

# release the camera and close all windows
#cap.release()
#cv2.destroyAllWindows()
