"""
print(" He said＼"Python is easy＼" ")

## escape sequence
\n 문자열 안에서 줄바꿈
\t 문자열 사이에 탭 간격을 줄 때 사용
\\ 문자 \ 를 그대로 표현
\' 작은따옴표를 그대로 표현
\" 큰따옴표를 그대로 표현


## Indexing & Slicing

movie = “Ocean’s Eleven”
print(movie[0 : 1])
O
print(movie[0 : 5])
Ocean
print(movie[1 : 7])
cean’s

string = "홀짝홀짝홀짝"
print(string[:: 2])
print(string[:: 3])

## 문자열 치환 (.replace)
phone = "031-400-5538"
phone1 = phone.replace("-", " ")
031 400 5538
phone1 = phone.replace("-", "")
0314005538

## 문자열 분리 (.split)
url = "http://hanyang.kr"
url_split = url.split(".")
print(url_split)
['http://hanyang', 'kr']
print( url_split[-1] )
kr


"""
#string = "홀짝홀짝홀짝"
#print(string[:: 2])
#print(string[:: 3])

#license_plate = "48구 4446"
#print(license_plate[:: -1])
#print(license_plate[-1:-4: -1])
#
#phone = "031-400-5538"
#phone1 = phone.replace("-", " ").replace(" ", "-")
# print(phone1)
#
#phone1 = phone.replace("-", "")
# print(phone1)
#
#url = "http://hanyang.kr"
# print(url.split(".")[-1])
#
#url_split = url.split(".")
#
# print(url_split)
# print(url_split[-1])
#
# for year in range(1979, 2019):
#    filename = f'LHF.{year}.nc'
#    filename = "LHF." + str(year) + ".nc"
#
#    print(filename)
#
#print(" He said \"Python is easy\" ")
#print("안녕하세요. \n파이썬은 재미 있나요?\n안녕히 계세요…")
# s = """안녕하세요.
# 파이썬은 재미 있나요?
# 안녕히 계세요…"""
# print(s)
#movie = "Ocean’s Eleven"
#print(movie[1: 7])

#string = "age"
#age = 25
#print("My %s is %d" % (string, age))
#
#score = 90.3854
#
#print("My math score is %.1f" % (score))
#print("My math score is %.2f" % (score))
#print("My math score is %.3f" % (score))
#
