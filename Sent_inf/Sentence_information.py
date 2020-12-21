import re

# 형량정보 끌어오기
for i in range(1, 158):
    f = open("C:/Users/SAMSUNG/PycharmProjects/Web_Crawling/" + str(i) + ".txt", 'rt', encoding='UTF-8')
    lines = f.read().splitlines()
    data = ' '
    lines = list(filter(None, lines))
    for line in lines:
        data = data + line

    sent_inf = re.search('주    문(.*)이    유', data)
    # print(sent_inf)
    if sent_inf == None:
        sent_inf = re.search('주       문(.*)이    유', data)
        if sent_inf == None:
            sent_inf = re.search('주문(.*)이유', data)
            if sent_inf == None:
                sent_inf = re.search('주문(.*)이    유', data)
                if sent_inf == None:
                    sent_inf = re.search('주         문(.*)이    유', data)
                    if sent_inf == None:
                        sent_inf = re.search('주   문(.*)이   유', data)

    sent_inf = sent_inf.group(1)
    f = open('sent_inf_' + str(i) + '.txt', 'w', -1, 'utf-8')
    f.write(sent_inf)
    f.close()