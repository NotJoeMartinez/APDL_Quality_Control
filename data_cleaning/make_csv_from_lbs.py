import json, os, re, csv

def main():
        json_path = 'final_bonded_unbonded.json'
        json_dict = make_dict(json_path)
        make_csv(json_dict)


def make_dict(json_path):

    label_dict = {}
    with open(json_path, "r") as read_json:
        data = json.load(read_json)
        for i in data:
            try:
                labels = i['annotations'][0]['result'][0]['value']['choices']
                image_name = i['file_upload']
                try:
                    image_name = re.search('^[^_]+(?=_)', image_name).group()
                except AttributeError:
                    pass

                if image_name.endswith(".jpg") == False:
                    image_name = image_name + ".jpg"

                label_dict.update({image_name:labels})
            except IndexError:
                pass
        return label_dict
                
def make_csv(json_dict):
    with open("data.csv", mode='w+') as writer:
        for key, value in json_dict.items():
            data_writer = csv.writer(writer, delimiter=',', quotechar='"')
            data_writer.writerow([key,value])






if __name__ == '__main__':
        main()