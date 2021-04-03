import cv2

def open_and_detect():
    save_path = '../test.avi'
    
    cap = cv2.VideoCapture(0)
    video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    video_size   = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(video_size)
    
    writer = cv2.VideoWriter(save_path, video_FourCC, 20.0, (640,480))
    
    while True:
        ret, frame = cap.read()
        writer.write(frame)
        
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    open_and_detect()