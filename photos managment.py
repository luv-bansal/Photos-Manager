import numpy as np
import os
import cv2 as cv
import face_recognition
import shutil
from sklearn.cluster import KMeans
pictures=r"C:\ML\google photos managment\photos"
faces=[]
tolerance=0.46
for filename in os.listdir(pictures):
    print(filename)
    image=face_recognition.load_image_file(f"{pictures}/{filename}")
    encoding_known=face_recognition.face_encodings(image)
    if encoding_known is not None:
        for i in range(len(encoding_known)):
            faces.append(encoding_known[i])
faces1=np.array(faces,dtype='float')
kmeans=KMeans(n_clusters=5)
kmeans.fit(faces1)
names=kmeans.fit_predict(faces1)
names=list(names)
for filename in os.listdir(pictures):
    image=face_recognition.load_image_file(f"{pictures}/{filename}")
    location=face_recognition.face_locations(image,model='hog')
    encoding=face_recognition.face_encodings(image,location)
    
    if encoding is not None:
        for face_enc, face_loc in zip(encoding,location):
            result=face_recognition.compare_faces(faces,face_enc,tolerance)
            if True in result:
                print('match found :')
                if not (os.path.exists(fr"C:\ML\project\identification\{names[result.index(True)]}")):
                    os.makedirs(fr"C:\ML\project\identification\{names[result.index(True)]}")
                if not (os.path.exists(fr"C:\ML\project\identification\{names[result.index(True)]}\{filename}")):
                    shutil.copy(fr"{pictures}/{filename}",fr"C:\ML\project\identification\{names[result.index(True)]}\{filename}")

                #cv.putText(image,'match found :',(face_loc[3],face_loc[0]),(0,255,0),5,cv.LINE_AA)
