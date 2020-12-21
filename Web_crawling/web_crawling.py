from bs4 import BeautifulSoup
from selenium import webdriver

baseurl = 'https://casenote.kr/search/?q=%ED%8F%AD%ED%96%89&sort=1&cc=3&ct=2&pr=0&pf=&pt=&p='

# 20개의 페이지 링크 저장할 리스트
link = []

# casenote에서 20페이지까지의 url
for i in range(1, 21):
    url = baseurl + str(i)
    link.append(url)

# 200개의 페이지 링크 저장할 리스트
case_link = []

for j in range(len(link)):
    driver = webdriver.Chrome()
    driver.get(link[j])
    html = driver.page_source
    soup = BeautifulSoup(html, features="lxml")
    r = soup.select('.searched_item')
    for k in r:
        case_link.append("https://casenote.kr" + k.a.attrs['href'])
    driver.close()

count = 1
# 판결문 내용 불러와서 텍스트 파일에 저장
for l in range(len(case_link)):
    driver = webdriver.Chrome()
    driver.get(case_link[l])
    html = driver.page_source
    soup = BeautifulSoup(html, features="lxml")
    title = soup.title.string.rstrip(" - CaseNote")
    caseurl = case_link[l]
    if "노" in title:
        driver.close()
        continue
    elif "도" in title:
        driver.close()
        continue
    else:
        title_1 = soup.find(style='display:inline-block; max-width: 700px;').get_text()
        print(title_1)
        content = soup.find(class_='editor').get_text()
        with open(str(count) + '.txt', 'w', -1, 'utf-8') as f:
            f.write(caseurl)
            f.write("\n")
            f.write(title_1)
            f.write(content)
            count = count + 1
        driver.close()





