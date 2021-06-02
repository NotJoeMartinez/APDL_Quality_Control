import os
import datetime as dt



""" Finds and returns the most recent model in the models directory """ 
def find_most_recent(directory):
  now = dt.datetime.now()
  dir_list = os.listdir(directory)
  datetimes = []
  for x in dir_list:
    dir_dt = dt.datetime.strptime(x, '%m_%d_%I:%M:%S%p')
    datetimes.append(dir_dt)

  most_recent = max(dt for dt in datetimes if dt < now)
  mr = most_recent.strftime("%m_%d_%-I:%M:%S%p")
  return mr

