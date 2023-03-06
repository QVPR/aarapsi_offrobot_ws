#!/usr/bin/env python3
import time
import rospy
import sys
import traceback

class Timer:
    def __init__(self,rospy_on=False):
        self.points = []
        self.rospy_on = rospy_on
        self.add_bounds = False

    def add(self):
        self.points.append(time.perf_counter())
    
    def addb(self):
        self.add_bounds = True

    def show(self, name=None, thresh=0.001):
        times = []
        for i in range(len(self.points) - 1):
            this_time = abs(self.points[i+1]-self.points[i])
            if this_time < thresh:
                this_time = 0.0
            times.append(this_time)
        if self.add_bounds and len(self.points) > 0:
            times.append(abs(self.points[-1] - self.points[0]))
        string = str(times).replace(' ','')
        if not (name is None):
            string = "[" + name + "] " + string
        self.print(string)
        self.clear()

    def clear(self):
        self.points[:] = []
        self.add_bounds = False

    def print(self, string):
        if self.rospy_on:
            try:
                rospy.loginfo(string)
                return
            except:
                pass
        print(string)

def formatException():
    # https://www.adamsmith.haus/python/answers/how-to-retrieve-the-file,-line-number,-and-type-of-an-exception-in-python
    exception_type, e, exception_traceback = sys.exc_info()
    filename = exception_traceback.tb_frame.f_code.co_filename
    line_number = exception_traceback.tb_lineno
    traceback_list = traceback.extract_tb(exception_traceback)
    traceback_string = ""
    for c, i in enumerate(traceback_list):
        traceback_string += "%s [%s]" % (str(i[2]), str(i[1]))
        if c < len(traceback_list) - 1:
            traceback_string += " >> "
    return "Exception Caught.\n\tDetails: %s %s\n\tFile %s [Line %s]\n\tTrace: %s" \
        % (str(exception_type), str(e), str(filename), str(line_number), traceback_string)