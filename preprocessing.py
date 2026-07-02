from read_data import read_csv

df = read_csv()

data = []

for row in df['text']:
    data.append(row.lower())

print(data[:2])