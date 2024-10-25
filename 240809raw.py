import pandas as pd
import xlwings as xw

# 엑셀 파일 열기 및 데이터프레임으로 변환
file_path = 'C:/Users/shinsegaeDF/Downloads/240809on.xlsx'
book = xw.Book(file_path)
sheet = book.sheets[0]
df = sheet.used_range.options(pd.DataFrame, index=False).value

# CSV 파일로 저장
df.to_csv('240809on.csv', index=False)

# 엑셀 파일 열기 및 데이터프레임으로 변환
file_path = 'C:/Users/shinsegaeDF/Downloads/240809off.xlsx'
book = xw.Book(file_path)
sheet = book.sheets[0]
df = sheet.used_range.options(pd.DataFrame, index=False).value

# CSV 파일로 저장
df.to_csv('240809off.csv', index=False)