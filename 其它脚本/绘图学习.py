import matplotlib.pyplot as plt


## 折线图绘制
x = [1,2,3,4,5]
y = [1,2,3,4,5]
x1 = [i for i in range(1,11)]
y1 = [i for i in range(1,20,2)]

plt.plot(x,y,x1,y1)
plt.xlabel("heng")
plt.ylabel("纵坐标")
plt.legend(["x1",'x2'])
plt.title('first')
plt.show()