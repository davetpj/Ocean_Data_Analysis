# for문과 range 함수

# for year in range(1979, 2019):
#     for month in range(1, 13):
#         # filename = f'LHF.{year}.nc'
#         if month < 10:
#             filename = "LHF." + str(year) + "0" + str(month) + ".nc"
#         else:
#             filename = "LHF." + str(year) + str(month) + ".nc"
#         print(filename)

# .zfill(2) 이렇게 하면 두 자리로 채워줌.

for year in range(1961, 2020):
    for month in range(1, 13):

        filename = f"LHF{str(year)}{str(month).zfill(2)}.nc"
        print(filename)

# 함수에서 숫자로 매개변수의 이름을 시작 할 수 없음.


# def mySum(number1, number2):
#     sum = number1+number2

#     return sum


# print(mySum(1, 5))
