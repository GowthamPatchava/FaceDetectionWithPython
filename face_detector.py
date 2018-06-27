#importing the cv2(computer vision)library
import cv2


#Linking the haarcascade_frontalface_default.xml file to our cascade classfier
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#Read the image file to be processed and detected
img=cv2.imread("charlie.jpg")


#Converting the given Image from BGRscale(BLUE,GREEN,RED Band's) to GRAY Scale
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#Detecting The face Cascades and storing in a variable
faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5)


#Representing the Detected Face with Recatangular Shaped Box
#255-is the value of color GREEN used for representating box border 
for x,y,w,h in faces:
    img=cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)
    
#specifying the type of faces and showing them    
print(type(faces))
print(faces)

#Reshaping the image in order to fit acc. to Display Resolution
resized=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))


#imshow function is used to display the image
cv2.imshow("Gray",img)

#The Duration for the window with Detected Face to be shown
cv2.waitKey(0)

#To close all the output windows
cv2.destroyAllWindows()





















































