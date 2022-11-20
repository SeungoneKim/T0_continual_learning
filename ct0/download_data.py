import gdown

train_files_url = "https://drive.google.com/drive/folders/1pzRPZRaklvZ3etxGt0cE5SJBhZ-HAxZ5?usp=share_link"
t0_eval_files_url = "https://drive.google.com/drive/folders/1kRo95ay_WqCFD43jRAqTi2of0yogvLN2?usp=share_link"

wiki_auto_url = "https://drive.google.com/drive/folders/1w_b2V9-Dasn_mYMkaXTGO2xSXxX9IZj-?usp=share_link"
asset_url = "https://drive.google.com/drive/folders/1pLzqIG8KwEMRVE-NOCyx9EpiP6jmlkWD?usp=share_link"
gigaword_url = "https://drive.google.com/drive/folders/1l8M8pgChuhhghAMiAx2bFl1Pg2a1Ui9W?usp=share_link"
covid_qa_url = "https://drive.google.com/drive/folders/1JZPOZ2C_SFY_TxGCs99PYeD4S7uYTnmG?usp=share_link"
haiku_url = "https://drive.google.com/drive/folders/1QMcHJPkyKbE9qGConjLc-YotqpDBVlpD?usp=share_link"
eli5_url = "https://drive.google.com/drive/folders/19KmYy2MGByFw8Eza6dpxrgi7EoQyOjrT?usp=share_link"
esnli_url = "https://drive.google.com/drive/folders/1vRnBb9RWhV3D5jYMtN8riTU2pRb2pepe?usp=share_link"
empathetic_dialogues_url="https://drive.google.com/drive/folders/1Kh71vItcoqcBZ7XpdOPYxDB69uMXryIr?usp=share_link"
twitter_url = "https://drive.google.com/drive/folders/1ki0e4bZsc9J0XYUAjVgrkiPSXX0oULot?usp=share_link"


# eval_files = [wiki_auto_url,asset_url,gigaword_url,covid_qa_url,eli5_url,haiku_url,esnli_url,empathetic_dialogues_url,twitter_url,t0_eval_files_url]
# gdown.download_folder(train_file_url, quiet=True, remaining_ok=True)
# for url in eval_files:
#     gdown.download_folder(url, quiet=True, remaining_ok=True)
gdown.download_folder(esnli_url, quiet=True, remaining_ok=True)