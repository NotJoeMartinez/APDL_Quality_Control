import os
import re
import sqlite3

# makes list of file paths
data_paths = []
for root, dirs, files in os.walk("dataset"):
   for name in files:
      data_paths.append(os.path.join(root, name))


con = sqlite3.connect('dataset.db')
cur = con.cursor()

print("makeing tables")
# make tabels
for file_path in data_paths:
    parent_dir = re.findall("\/(\w*)\/", file_path)
    # Make parent dir column if not in
    cur.execute("CREATE TABLE IF NOT EXISTS {}(label STRING, filename STRING);".format(parent_dir[0]))
con.commit()

print("populating tabels")
# populate tabels
for file_path in data_paths:
    parent_dir = re.findall("\/(\w*)\/", file_path)
    file_name = re.findall("\/.*\/(.*\.png)",file_path)
    try:
        cur.execute("INSERT INTO {}(label, filename) VALUES('{}','{}');".format(parent_dir[0], parent_dir[0], file_name[0]))
    except IndexError:
        pass

print("saved tabel")
con.commit()    
con.close()