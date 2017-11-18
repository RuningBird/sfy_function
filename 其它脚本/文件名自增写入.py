num = 0
filepath = "/datas/"
with open(filepath + "n1.txt", 'r') as f:
    ns = f.readline()
    num = int(ns)
    print(num)

with open(filepath + "n1.txt", 'w') as f:
    tmp = num + 1
    f.write(str(tmp))
new_file = filepath + "file_" + str(num) + ".txt"
print(new_file)

with open(new_file, 'w') as f:
    tmp = num + 1
    f.write(str(tmp))
