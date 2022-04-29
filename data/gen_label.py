import sys
sys.path.append('../src')
from utils import read_json, write_json
import os

data = read_json('val.json')
# a = ".." + data[0][36:]
for i in range(len(data)):
    data[i] = ".." + data[i][36:]

print(data)
write_json("val.json", data)