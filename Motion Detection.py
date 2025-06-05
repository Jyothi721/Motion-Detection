import cv2,os,numpy
size=4
haar_cascade='haarcascade_frontaleface_document.xml'
datasets='datasets'
print('Training...')
(images,lables,names,id)=([],[],[],0)
for (subdir,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        name[id]=subdir
        subjectpath=os.join.path(datasets)
        for filename in os.listdir(subjectpath):
            path=subject+'/'+filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(labels))
            id+=1
(width,height)=(130,100)
(images,lables)=[numpy.array(lis) for lis in [images,lables]]
model=cv2.face.FisherFaceRecognizer_create()
model.train(images,lables)
face_cascade=cv2.CascadeClassifier(haar_cascade)
webcam=cv2.VideoCapture(1)
cnt=0
while True:
    (_,im)=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,width,height)
        prediction=model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
        if prediction[1] < 800:
            cv2.putText(im,'%s - %o.f'%(names[prediction[0]],prediction[1],(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2))
            print(prediction[0])
            cnt=0
        else:
            cnt +=1
            cv2.putText(im,'unkown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),3)
            if cnt > 100:
                print("Unkown Person")
                cv2.imwrite("input.jpg",im)
                cnt=0
    cv2.imshow('OpenCV',im)
    key=cv2.waitKey(10)
    if key == 27:
      break
webcam.release()
cv2.destroyAllWindows()
  
