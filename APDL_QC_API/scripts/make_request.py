
import os, subprocess, sys
subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)

def main():
    img_path = sys.argv[1]

    try:
        server_ip = sys.argv[2] 
    except IndexError:
        server_ip = "127.0.0.1:5000"
        pass

    data_paths = []
    for root, dirs, files in os.walk(img_path):
        for img in files:
            data_paths.append(f"{root}/{img}")

    for data_path in data_paths:
        subprocess.run(f"curl -X POST -F 'image=@{data_path}' http://{server_ip}/get-labels | json_pp", shell=True)
    

if __name__ == '__main__':
    main()