num = 0
with open("/datas/n.txt", 'r') as f:
    ns = f.readline()
    num = int(ns)
    print(num)

with open("/datas/n.txt", 'w') as f:
    tmp = num + 1
    f.write(str(tmp))
