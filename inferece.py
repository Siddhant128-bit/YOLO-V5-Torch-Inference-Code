import torch
import torchvision
from PIL import Image
import cv2
import numpy as np 


color_list={
    'monitor':(255,0,0),
    'laptop':(0,255,0),
    'mouse':(0,0,255),
    'keyboard':(160,32,240),
    'cell phone':(255, 165, 0)
}

def initial_reference():
    # Load YOLO v5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Load input image
    img = Image.open('D:\Projects\yolo_V5_utilities_weights_\\test.jpg')

    # Perform object detection on input image
    results = model(img)

    # Print detected objects and their confidences
    print(results.pandas().xyxy[0])


def inference_image_draw(model,img):
    class_count={}
    thres_conf=0.40
    primary_classes_list=['tv','laptop','mouse','keyboard','cell phone']
    if img=='False':
        img=cv2.imread(input('Enter path of image to test: '))

    img=cv2.resize(img,(500,500))
    results=model(img)
    # Get the bounding box coordinates and draw them on the image
    boxes = results.pandas().xyxy[0]
    for i in primary_classes_list:
        class_count[i.replace('tv','monitor')]=len(boxes.loc[(boxes['name'].str.contains(i)) & (boxes['confidence']>=thres_conf)])
    
    info_dump=np.zeros((500,500,3),dtype=np.uint8)
    info_dump=cv2.rectangle(info_dump,(50,50),(450,450),(255,255,255,-1))
    
    for _, box in boxes.iterrows():
        if box['name'].lower() in primary_classes_list and box['confidence']>=thres_conf:
            box['name']=box['name'].replace('tv','monitor')    
            x1, y1, x2, y2 = box[['xmin', 'ymin', 'xmax', 'ymax']].values
            color_val=color_list[box['name']]
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_val, 1)
            cv2.putText(img,box['name'],(int(x1)-5,int(y1)-5),cv2.FONT_HERSHEY_SIMPLEX,1,color_val,2)
    
    y_increase=50
    for key,value in class_count.items():
        cv2.putText(info_dump,f'{key}: {value}',(100,100+y_increase),cv2.FONT_HERSHEY_SIMPLEX,1,color_list[key],2)
        y_increase+=50
    
    return img,class_count,info_dump


def inference_video_draw(model):

    cap = cv2.VideoCapture(input('Enter path of the video: '))

    # Loop through video frames
    while True:
        # Read frame from video
        ret, frame = cap.read()

        # If there's no frame, break the loop
        if not ret:
            break

        # Display the frame
        frame,count,info_dump=inference_image_draw(model,frame)
        cv2.imshow('frame', frame)
        cv2.imshow('Count',info_dump)

        # Wait for 25ms, then check if the 'q' key was pressed to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release video file and close window
    cap.release()
    cv2.destroyAllWindows()    

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
choice=int(input('Enter 1 to infer with image and 2 to infer with video: '))
if choice==1:
    img,count_info,info_dump=inference_image_draw(model,'False')
    cv2.imshow('Output',img)
    cv2.imshow('Count_Infor',info_dump)
    cv2.waitKey(0)
else:
    inference_video_draw(model)