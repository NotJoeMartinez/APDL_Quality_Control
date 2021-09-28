import sqlite3, csv, re, subprocess, hashlib

def make_aug_table():
     # make tables
    con = sqlite3.connect('apdl.db')
    cur = con.cursor()

    #     list_data=[og_filepath,new_name,aug_function,angle, ]
    sql = '''CREATE TABLE if NOT EXISTS 
                datasets (shasum text, 
                og_filepath text, 
                new_filepath text,
                aug_function text, 
                angle text,
                dataset_name text);'''

    cur.execute(sql)
    con.commit()
    con.close()

def execute_sql(shasum,og_filepath, new_filepath, aug_function, angle, dataset_name):
    print(f"{shasum}, {og_filepath}, {new_filepath}, {aug_function} ,{angle}, {dataset_name}")
 
    con = sqlite3.connect('apdl.db')
    cur = con.cursor()
    cur.execute("INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?) ", (shasum, og_filepath,new_filepath,aug_function,angle, dataset_name))
    con.commit()
    con.close()


def get_shasum(filepath):
    shasum = subprocess.check_output(f"shasum ../{filepath}", shell=True)
    shasum = shasum.decode()
    foo = re.search("^(.*) \.\.", shasum).group(1)
    return foo

def add_csv_todb():
    csv_path = "csvs/augmentations/09_27_21_11_1632795113.csv"
    dataset_name = re.search("\/(.*)\.csv$",csv_path).group(1)

    with open(csv_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split(",")
            og_filepath = row[0]
            new_filepath = row[1]
            aug_function = row[2]
            angle = str(row[3])
            shasum = get_shasum(og_filepath)
            execute_sql(shasum, og_filepath,new_filepath,aug_function,angle,dataset_name)


add_csv_todb()