import numpy as np
import os
import cv2 as cv
import face_recognition
import shutil
from sklearn.cluster import DBSCAN
#from sklearn.cluster import KMeans
#from sklearn.cluster import AgglomerativeClustering
pictures=r"/photos"
faces=[]
tolerance=0.5
for filename in os.listdir(pictures):
    print(filename)
    image=face_recognition.load_image_file(f"{pictures}/{filename}")
    encoding_known=face_recognition.face_encodings(image)
    if encoding_known is not None:
        for i in range(len(encoding_known)):
            faces.append(encoding_known[i])
faces1=np.array(faces,dtype='float')
#kmeans=KMeans(n_clusters=5)
#kmeans.fit(faces1)
#names=kmeans.fit_predict(faces1)
#harc=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='ward')
#names=harc.fit_predict(faces1)
dbscan=DBSCAN(eps=0.4,min_samples=6,metric="euclidean")
dbscan.fit(faces1)
names=dbscan.labels_
names=list(names)
q=0
for filename in os.listdir(pictures):
    image=face_recognition.load_image_file(f"{pictures}/{filename}")
    location=face_recognition.face_locations(image,model='hog')
    encoding=face_recognition.face_encodings(image,location)
    
    if encoding is not None:
        for face_enc, face_loc in zip(encoding,location):
          image=face_recognition.load_image_file(f"{pictures}/{filename}")
          result=face_recognition.compare_faces(faces,face_enc,tolerance)
          if True in result:
              q+=1
              print('match found :')
              cv.rectangle(image,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(0,255,0),4)
              if not (os.path.exists(fr"/identification/{names[result.index(True)]}")):
                  os.makedirs(fr"/identification/{names[result.index(True)]}")
              if not (os.path.exists(fr"/identification/{names[result.index(True)]}/{filename}")):
                  #shutil.copy(fr"{pictures}/{filename}",fr"/identification/{names[result.index(True)]}/{filename}")
                  cv.imwrite(fr"/identification/{names[result.index(True)]}/{filename}",image)

                #cv.putText(image,'match found :',(face_loc[3],face_loc[0]),(0,255,0),5,cv.LINE_AA)
