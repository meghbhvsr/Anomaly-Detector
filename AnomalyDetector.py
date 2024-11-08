import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    # Used the Isolation Forest Algorithm, which is great for detecting anomalies when distribution or seasonality of data changes over time.
    # The isolation forest alghorithm is a machine learning algorithm for anomaly detection. It's an unsupervised learning algorithm that recognizes anomalies by isolation any outliers in the data.
    # This algorithm is based on the decision tree algorithm it randomly partitions at each step and creates shorter paths in the trees for anomaly points, which helps us tell the difference between regular points
    # and anomalies. 
    # Use sliding window to only look at 50 data points at a time, in order to get a more accurate model, and anomally detection
    def __init__(self, windowSize=50):
        if not isinstance(windowSize, int) or windowSize <= 0:
            raise ValueError("windowSize must be a positive integer.")
        self.windowSize = windowSize
        # Use a queue for the window, so we can remove from the front, and add to the end
        self.window = deque(maxlen=windowSize)
        self.model = IsolationForest(contamination=0.05)
    
    def updateModel(self):
        data = np.array(self.window).reshape(-1, 1)
        self.model.fit(data)
    def detect(self, newValue):
        if not isinstance(newValue, (int, float)):
            print("Invalid data type for newValue; must be an integer or float.")
            return False
        self.window.append(newValue)
        if len(self.window) == self.windowSize:
            # if our current window has a length of 50, we update the model, by training it, and check if new value is an anomaly
            self.updateModel()
            isAnomally = self.model.predict([[newValue]])[0] == -1 #This indicates wether the last value added is an anomaly
            return isAnomally
        return False

#Creates random data points, taking into account seasonal, noise and anomalies
def dataStream():
    t = 0
    while True:
        seasonal = 10 * np.sin(2 * np.pi * t / 50) # Creates a sine wave with amplitude of 10 while (2 * np.pi * t / 50) is the frequency
        noise = random.gauss(0, 1) # Random value from gaussian dist with mean 0, and standard deviation of 1
        anomaly = 50 if random.random() < 0.01 else 0 #1 percent chance for occasional anomalies
        yield seasonal + noise + anomaly
        t += 1

detector = AnomalyDetector(windowSize=50)
stream = dataStream()

streamData = []
anomalies = []
#plotting the graph
figure, axis = plt.subplots()
line, = axis.plot([], [], lw=2)
anomalyPoints, = axis.plot([], [], 'ro')
axis.set_xlim(0, 100)
axis.set_ylim(-20, 60)
axis.set_title("Real-Time Data Stream Anomaly Detection with Isolation Forest")
axis.set_xlabel("Time")
axis.set_ylabel("Value")

def init():
    line.set_data([], [])
    anomalyPoints.set_data([], [])
    return line, anomalyPoints

# Update the plot, based on if the point is an anomaly or not, point will be red, if it is an anomaly
def update(frame):
    global streamData, anomalies
    newVal = next(stream) #Gets random value from data stream
    streamData.append(newVal)

    #Check is new value is an anomaly
    if detector.detect(newVal):
        anomalies.append((len(streamData)-1, newVal))
    
    #Sets the points on the graph
    line.set_data(range(len(streamData)), streamData)

    #Sets the anomaly points on the graph - show up as red
    anomalyIndexes, anomalyValues = zip(*anomalies) if anomalies else ([], [])
    anomalyPoints.set_data(anomalyIndexes, anomalyValues)

    if len(streamData) > 100:
        axis.set_xlim(len(streamData) - 100, len(streamData))

    return line, anomalyPoints

ani = animation.FuncAnimation(figure, update, init_func=init, frames=500, interval=100, blit=True)
plt.show()