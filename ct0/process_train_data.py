import json

file_name = input('Write the directory to file name : ')

data = []

json_data = [json.loads(line) for line in open(file_name,'r')]
for t in json_data:
    src = t['translation']['en1']
    tgt = t['translation']['en2']
    data.append({"en1":src,"en2":tgt})

with open(file_name.replace('.json','')+"_preprocessed.json","w") as ff:
    for item in data:
        ff.write(json.dumps({"en1": item["en1"], "en2": item["en2"]}, ensure_ascii=False) + "\n")
        