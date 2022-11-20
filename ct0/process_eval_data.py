import json

file_name = input('Write the directory to file name : ')

data = []
with open(file_name,'r') as f:
    json_data = json.load(f)
    for src,tgt in zip(json_data["src"],json_data["tgt"]):
        data.append({"en1":src,"en2":tgt})

with open(file_name.replace('.json','')+"_preprocessed.json","w") as ff:
    # for item in data:
    #     ff.write(json.dumps({"en1": item["en1"], "en2": item["en2"]}, ensure_ascii=False) + "\n")
    json.dumps()
        