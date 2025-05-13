import cv2, queue as Queue, threading, time


is_frame = True
# bufferless VideoCapture


class VideoCaptureQ:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = Queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                global is_frame
                is_frame = False
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except Queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
