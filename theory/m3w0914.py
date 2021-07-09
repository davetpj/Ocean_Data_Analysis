# numpy array
import numpy
temperature = [22, 22.5, 23, 22, -999, 22.5, -999, 24, 26, 27, -999, 26.5]

temp = numpy.array(temperature)
new_Temp = (temp != -999)
print(new_Temp)
new_Temp = temp[new_Temp]
print(new_Temp)
mean = numpy.mean(new_Temp)
print(mean)

mean_temp = 0
new_Temp = []
for temp in temperature:
    if temp != -999:
        new_Temp.append(temp)
        mean_temp = temp + mean_temp
    else:
        None
    new_mean = mean_temp/len(new_Temp)
print(new_Temp)
mean = numpy.mean(new_Temp)
print(mean)
print(new_mean)


# # append()
# a = [1, 2, 3]
# a.append(4)
# print(a)
# # extend()
# a = [1, 2, 3]
# a.extend([4, 5, 6])
# print(a)
# # insert()
# a = [2, 4, 5]
# a.insert(0, "Hi")
# a.insert(-1, "Nice")
# print(a)
# # remove()
# a = ["BMW", "BENZ", "VOLKSWAGEN", "AUDI"]
# a.remove("BMW")
# print(a)
# # pop()
# a = [1, 2, 3, 4, 5]
# a.pop()
# print(a)
# a = [1, 2, 3, 4, 5]
# a.pop(2)
# print(a)
# # index()
# a = ["abc", "def", "ghi"]
# print(a.index("def"))
# # count()
# a = [1, 100, 2, 100, 3, 100]
# print(a.count(100))
# print(a.count(200))
