
import cv2
import time
from yolo_detector import YoloDetector
from tracker import Tracker
from collections import deque
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import torch


TIME_INTERVAL=5

def drawText(frame, counter_in, counter_out, fps):

    cv2.putText(frame, "IN", (604, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 5)
    cv2.putText(frame, "OUT", (604, 104), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 5)
    cv2.putText(frame, "FPS", (604, 154), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 5)
    cv2.putText(frame, f"{counter_in}", (754, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 5)
    cv2.putText(frame, f"{counter_out}", (754, 104), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 5)
    cv2.putText(frame, f"{'%.2f'%(fps)}", (750, 154), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 5)


    cv2.putText(frame, "IN", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    cv2.putText(frame, "OUT", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    cv2.putText(frame, "FPS", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (55, 55, 255), 5)
    cv2.putText(frame, f"{counter_in}", (750, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    cv2.putText(frame, f"{counter_out}", (750, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    cv2.putText(frame, f"{'%.2f'%(fps)}", (750, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (55, 55, 255), 5)



def drawResults(cIn, cOut, tStamps,inSum,outSum):
    plt.scatter(tStamps[1:], cIn[1:], label="KA RASKRSNICI", color="red", marker="*", s=30)
    plt.scatter(tStamps[1:], cOut[1:], label="IZ RASKRSNICE", color="green", marker="*", s=30)


    plt.xlabel("vreme u sekundama")
    plt.ylabel("broj vozila")
    plt.title("Procena gustine saobracaja")
    plt.legend()

    # Podesavanje koraka za x i y ose
    x_step = TIME_INTERVAL  # Primer koraka za x osu
    y_step = 1  # Primer koraka za y osu
    plt.xticks(range(int(min(tStamps)+TIME_INTERVAL), int(max(tStamps) + TIME_INTERVAL) + 1, x_step))
    plt.yticks(range(int(min(min(cIn), min(cOut)) ), int(max(max(cIn), max(cOut)) + TIME_INTERVAL) + 1, y_step))

    plt.show()


def main():

    Tk().withdraw()
    VIDEO_PATH = askopenfilename(title="Izaberite video fajl",filetypes=[("Video fajlovi", "*.mp4 *.avi *.mov")])
    MODEL_PATH = askopenfilename(title="Izaberite model",filetypes=[("YOLO model","*.pt")])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = YoloDetector (model_path= MODEL_PATH, confidence=0.33)
    detector.model.to(device)

    tracker = Tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)

    fpsIn = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    timeStamps = []

    seconds = totalNoFrames / fpsIn
    minutes = int(seconds / 60)
    rem_sec = int(seconds % 60)
    numOfDots = seconds / TIME_INTERVAL
    flag =1


    if not cap.isOpened():
        print("Greska prilikom ucitavanja video snimka!")
        exit()

    points = [deque(maxlen=64) for _ in range(1000)]  # cuvanje pointa za svaki track
    counter_in = 0
    counterPart_In = 0
    counterPart_Out = 0
    counter_out = 0

    fpsSum = 0
    fpsNum = 0

    countersIn = []
    countersOut = []
    flag =0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()
        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)

        y_line = int(frame.shape[0] * 0.5)
        counter_line = ((0, y_line), (frame.shape[1], y_line))

        for tracking_id, bBox in zip(tracking_ids, boxes):
            cv2.rectangle(frame, (int(bBox[0]),int(bBox[1])),(int(bBox[2]),int(bBox[3])),(0,0,255),2)
            cv2.putText(frame,f"{str(tracking_id)}",(int(bBox[0]), int(bBox[1]-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)
            center_x = int((int(bBox[0]) + int(bBox[2])) / 2)
            center_y = int((int(bBox[1]) + int(bBox[3])) / 2)

            points[int(tracking_id)].append((center_x, center_y)) #apendujem pointe za svaki track
            cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

            start_point_x = points[int(tracking_id)][0][0]
            start_point_y = points[int(tracking_id)][0][1]
            cv2.circle(frame, (int(start_point_x), int(start_point_y)), 4, (255, 255, 255), -1)

            if center_y > counter_line[0][1] and start_point_y < counter_line[0][1]:
                counter_in += 1
                points[int(tracking_id)].clear()
            elif center_y < counter_line[0][1] and start_point_y > counter_line[0][1]:
                counter_out += 1
                points[int(tracking_id)].clear()

        timestamp= cap.get(cv2.CAP_PROP_POS_MSEC)
        if(timestamp>=TIME_INTERVAL*1000 *flag and flag<= numOfDots):
             countersIn.append(counter_in - counterPart_In)
             counterPart_In=counter_in
             countersOut.append(counter_out - counterPart_Out)
             counterPart_Out = counter_out
             timeStamps.append(TIME_INTERVAL*flag)
             flag += 1


        end_time=time.perf_counter()
        fps=1/(end_time-start_time)
        fpsSum +=fps
        fpsNum += 1

        drawText(frame,counter_in, counter_out, fps)

        cv2.line(frame, counter_line[0], counter_line[1], (0, 255, 0), 5)

        frame = cv2.resize(frame, (1270,720))
        cv2.imshow("Procena saobracaja",frame)

        key = cv2.waitKey(1) &0xFF
        if(key)==ord("q"):
            break

    drawResults(countersIn, countersOut, timeStamps,counter_in,counter_out)

    avgFps='%.3f'%(fpsSum/fpsNum)
    messagebox.showinfo("Analiza","Trajanje snimka: "+str(minutes)+" min, "+str(rem_sec)+"s"+"\n"+"Prosek obrade: "+str(avgFps)+" frejmova u sekundi")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
            main()