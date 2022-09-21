import cv2
import os
times=0
frameFrequency=1
outPutDirName=r''
if not os.path.exists(outPutDirName):
    os.makedirs(outPutDirName)
camera = cv2.VideoCapture(r'demo.wmv')
while True:
    times+=1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if times%frameFrequency==0:
        cv2.imwrite(outPutDirName + str(times).rjust(6,'0')+'.png', image[:,:,:])
        print(outPutDirName + str(times)+'.png')
print('overall')
camera.release()
