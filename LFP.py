import cv2
from PIL import Image
from base_model import ImageProcessing

class LiveFrameProcesser:

    def FrameInputProcesser():

        #print(torch.cuda.is_available())
        #print(torch.cuda.device_count())
        #print(torch.cuda.current_device())
        #print(torch.cuda.get_device_name(0))

        #processing_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        #mtcnn = MTCNN(device = 'cuda:0')
        #print('MTCNN Device:', mtcnn.device)

        name = ''
        rebase = False
        ip = ImageProcessing()

        if rebase:
             ip.chroma.client.reset()
             ImageProcessing().InitializeDB()             
             exit()
        
        cap = cv2.VideoCapture(1) #0 - 720p Cam / MacOS Continuity-Cam, #1 - 4K Cam / MacOS, #2 - OBS Virtual Cam

        if not cap.isOpened():
                print('Camera IO Exception')
                exit()
        
        while True:
                ret, frame = cap.read()

                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                deinverted_frame = cv2.flip(frame, 1)
                frame = deinverted_frame
                
                boxes, probabilities = ip.mtcnn.detect(frame)

                if boxes is not None:
                    for box, prob in zip(boxes, probabilities):
                        x, y, w, h = map(int, box)
                        confidence_score = prob
                        w //= 2
                        h //= 2
                        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
                        cv2.putText(frame, 'Confidence Score: {:.2f}%'.format(confidence_score), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
 
                        cv2.putText(frame, 'Person: {}'.format(name), (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        img_embedding = ip.Frame2Tensor(frame = frame)
                        data = ip.Tensor2Image(tensor = img_embedding)
                        name = data[0] if data is not None else ''

                        if cv2.pollKey() == ord('c'):
                            if data is not None:
                                linked_prof : Image = data[1]
                                linked_prof.show()


                cv2.imshow('Live Feed - Face Detection', frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

        return frame
    
def main():
    LiveFrameProcesser.FrameInputProcesser()
    #LiveFrameProcesser.FacialRecog()

main()