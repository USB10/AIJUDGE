#!/usr/bin/env python
# coding: utf-8

# In[37]:


# 판결문 예시1
s = '''서울남부지방법원 2017. 7. 21. 선고 2017고합50 판결 [폭행치사(인정된 죄명: 폭행)]  



사 건
2017고합50  폭행치사(인정된 죄명: 폭행) 


피고인
이◯◯  (69년생, 남) 


검사
유옥근(기소), 현동길(공판) 


변호인
변호사 윤희숙, 박승휘(각 국선) 


판결선고
2017. 7. 21.

주    문
피고인을 징역 1년 6월에 처한다.
이    유
범 죄 사 실
  피고인은 2017. 1. 28. 19:05경 서울 영등포구 경인로102길 10 ‘희망지원센터’ 앞 노상에서, 술에 취한 피해자 고◯◯(47세)이 피고인을 쫓아다니며 계속 욕설을 하자 화가 나 오른손 주먹으로 피해자의 안면부를 1회 가격하여 그 곳 바닥에 머리를 부딪치게  하는 등 폭행을 가하였다.
증거의 요지
생략
법령의 적용
1. 범죄사실에 대한 해당법조
   형법 제260조 제1항(징역형 선택)
양형이유
1. 법률상 처단형의 범위: 징역 1월 ~ 2년
2. 양형기준에 의한 권고형의 범위: 징역 2월 ~ 10월
  [유형의 결정]  폭력범죄 > 폭행범죄 > 제1유형(일반폭행)
  [권고영역 및 권고형의 범위] 기본영역, 징역 2월 ~ 10월
3. 선고형의 결정: 징역 1년 6월
  이 사건은 피고인이 피해자의 얼굴부위를 주먹으로 폭행한 사건으로 피고인이 행위로 인하여 발생한 결과로 피해자는 숭고한 생명을 잃게 되어 그 책임이 가볍지 않은  점, 피고인은 동종 폭력범죄로 처벌받은 전력이 여러 차례 있는 자로 다시 이 사건과  같이 중한 결과를 발생시킨 범행을 저지른 점, 피고인이 피해회복을 위하여 노력한 것으로 보이지 아니하고 피해자의 유족으로부터도 용서받지 못하고 있는 점 등을 고려하면 그 책임에 상응하는 처벌이 필요하다.
  다만, 피고인이 피해자를 1회 때린 것에 그쳤고 피해자가 쓰러진 후 더 이상 폭행이  계속되지는 않은 점, 피고인이 범행을 반성하고 있는 것으로 보이는 점 등을 피고인에  대한 유리한 정상으로 참작한다.
    그 밖에 피고인의 나이, 성행, 건강상태, 지능과 환경 등 기록과 변론에 나타난 여러 사정을 종합하여 양형기준의 상한을 일탈하여 주문과 같이 형을 정한다.  
무죄부분
1. 공소사실의 요지
  피고인은 판시 범죄사실과 같이 피해자를 폭행하여 그로 인하여 그 자리에서 피해자를 머리 손상(머리뼈 골절, 뇌지주막하출혈, 뇌실내출혈)으로 사망에 이르게 하였다.
2. 피고인 및 변호인의 주장 요지
  피고인은 피해자를 폭행할 당시 자신의 행위로 인하여 피해자가 사망에 이를지도 모른다는 점에 대한 예견가능성이 없었다.
3. 배심원 평결
  ○ 무죄: 만장일치
4. 판단
  이 부분 공소사실은 그 범죄사실의 증명이 없는 경우에 해당하므로 형사소송법 제325조 후단에 의하여 무죄를 선고하여야 할 것이나 이와 동일한 공소사실의 범위 내에  있는 판시 폭행죄를 유죄로 인정한 이상 달리 주문에서 무죄를 선고하지 않는다.
배심원 평결과 양형의견
1. 유·무죄에 대한 평결
 가. 폭행치사죄
  ○ 무죄: 만장일치
 나. 폭행죄
  ○ 유죄: 만장일치
2. 양형에 대한 의견
  ○ 징역 2년: 5명
  ○ 징역 10월: 2명
  이상의 이유로 피고인에 대한 이 사건을 그 희망에 따라 국민참여재판을 거쳐 주문과 같이 판결한다.     


재판장 

판사 

심형섭 



 

판사 

김지연 



 

판사 

이상언 




'''

# In[38]:


# 판결문 예시2
s2 = '''부산지방법원 2016. 7. 22. 선고 2016고단2905,  2016고단3176(병합) 판결 [상해, 폭행]  



사 건
2016고단2905,  2016고단3176(병합)  상해, 폭행 


피고인
A 


검사
소창범(기소), 성두경(공판) 


변호인
변호사 B(국선) 


판결선고
2016. 7. 22.

주    문
피고인을 징역 1년 6월에 처한다.
이    유
범 죄 사 실
 「2016고단2905」
1. 피고인은 2016. 4. 9. 01:15경 부산 사상구 C에 있는 'D주점'에서 별다른 이유 없이 옆좌석에 있던 피해자 E(47세)에게 욕설을 하며 시비를 걸다가 피해자가 주점 밖으로 나가자 피해자를 뒤따라가 주먹으로 얼굴을 1회 때려 바닥에 넘어뜨린 후 다시 발로 피해자의 얼굴, 배, 다리 등을 수회 걷어찼다.
  또한 피고인은 옆에서 이를 말리려 하던 피해자 E의 처인 피해자 F(여, 51세)을 손으로 밀어 바닥에 넘어뜨린 후 발로 피해자의 엉덩이 등을 수회 걷어찼다.
이로써 피고인은 피해자 E에게 약 6주간의 치료를 요하는 우측안와골절 및 우측삼각골골절 등의 상해를 가하고, 피해자 F을 폭행하였다.
 「2016고단3176」
2. 피고인은 2016. 5. 13. 02:10경 부산 사하구 G에 있는 'H' 주점 앞길에서 피해자 I(56세)이 위 주점 내에서 피고인에게 "중놈이 한 놈 와서 노래를 한다."고 말하였다는 이유로 화가 나 주먹과 발로 피해자의 머리, 얼굴 등을 수차례 때려 피해자에게 약 2주간의 치료가 필요한 결막하출혈 및 우안 안검반상출혈 등의 상해를 가하였다.
증거의 요지
 「2016고단2905」
1. 피고인의 법정전술
1. J, E, F에 대한 각 경찰진술조서
1. 상해진단서
 「2016고단3176」
1. 피고인의 법정잔술
1. I에 대한 경찰진술조서
1. 진단서
1. 수사보고(증거목록 순번 2, 6번)
법령의 젹용
1. 범죄사실에 대한 해당법조 및 형의 선택
   각 형법 제257조 제1항(상해의 점), 형법 제260조 제1항(폭행의 점), 각 징역형 선택
1. 경합범가중
   형법 제37조 전단, 제38조 제1항 제2호, 제50조(형과 범정이 가장 무거운 피해자 E에 대한 상해죄에 정한 형에 경합범가중)
양형의 이유
1. 법률상 처단형의 범위 : 징역 1월 ~ 10년 6월
2. 양형기준의 적용
  가. 각 상해죄
  [범죄유형의 결정] 폭력범죄 > 일반적인 상해 > 제1유형(일반상해)
  [특별양형인자] 없음
  [권고형의 범위] 기본영역, 징역 4월 ~ 1년 6월
  나. 폭행죄
  [범죄유형의 결정] 폭력범죄 > 일반적인 상해 > 제1유형(일반상해)
  [특별양형인자] 없음
  [권고형의 범위] 기본영역, 징역 4월 ~ 1년 6월
  다. 다수범 가중에 따른 최종 형량범위 : 징역 4월 ~ 2년 6월 10일
3. 선고형의 결정
  가. 1) 일반적으로 범죄가 발생하려면 다음 세가지 조건이 충족되어야 한다. 첫째, 범죄를 저지를 합리적인 동기가 있는 가해자가 있어야 한다. 둘째, 범죄자가 합리적으로 선택한 피해자가 있어야 한다. 셋째, 범죄 발생을 억제하는 적합한 보호요소가 없어야 한다. 그런테 최근에는 범죄자와 피해자 사이에 상관관계가 없거나, 범죄 동기가 없거나 범죄 동기를 일반인이 이해할 수 없는 불특정의 대상을 상대로 행해지는 범죄행위, 일명 '묻지마 범죄'가 급증하고 있다.
    2) 묻지마 범죄는 범죄자의 특성을 기준으로 할 때, 현실 불만형, 만성 분노형, 정신장애형으로 분류될 수 있다. 현실 불만형 범죄자들은 주로 사회에 불만이 있거나 자신의 처지를 비관해 묻지마 범죄를 저지른다. 만성 분노형 범죄자들은 다른 사람의 행동이나 의도를 곡해하는 특징이 있고, 특별한 이유 없이 재미로 범죄를 저지르기도 하며, 술을 마신 상태에서 폭력범죄를 저지르거나 상습 폭력범인 경우가 많다. 정신장애형 범죄자들은 정신과 치료 경험이 있는 경우가 많으며, 범행 당시 신체 건강이 양호하지 못하거나, 망상이나 환각에 사로잡혀 있는 경우도 있다.
    3) 묻지마 범죄는 단순히 같은 공동체의 구성원이라는 이유만으로 일면식도 없는 사람을 대상으로 폭력을 사용하여 분노를 표출함으로써 공동체의 구성원들에게 불안과 공포 및 공동체의 안전에 대한 불신을 초래한다. 국가라는 공동체가 구성되는 가장 근본적인 이유는 만인 대 만인의 투쟁으로부터 공동체 구성원들의 생명과 신체의 안전을 보장하려는 데 있다. 따라서 국가기관은 엄정한 법집행을 통해 묻지마 범죄자를 사회로부터 격리시킴과 동시에 묻지마 범죄는 어떠한 이유로도 용인될 수 없음을 공고히 할 필요가 있다.
  나. 그런데 기록에 의하면, 피고인은 자식이 없어 이혼하고 가족들과도 연락이 끊긴 후 오랫동안 외톨이로 지내면서 술만 먹으면 스스로에게 화가 나고 세상이 싫어져 폭력 범죄를 저질러 왔다고 진술한 점, 피고인은 폭력범죄로 다수의 형사처벌을 받은 전력이 있고, 폭력범죄로 집행유예기간 중일 뿐만 아니라 그 기간 중에 폭력범죄를 저질러 벌금형의 선처를 받았음에도 또다시 이 사건 각 범행을 저지른 점, 피고인은 아무런 이유 없이 판시 제1항의 각 범행을 저지른 점 등을 고려할 때, 피고인은 만성 분노형 범죄자로서 상당한 기간 동안 사회로부터 격리시킬 필요가 있어 실형의 선고가 불가피하되, 피고인의 자백, 피해자들의 처벌의사, 피고인의 연령, 성행, 환경, 범행 후의 정황 등을 종합적으로 고려하여 주문과 같이 형을 정한다.

 

판사 

이승훈 


'''

# In[39]:


# 문장 예시(뉴스)
sents = [
    '오패산터널 총격전 용의자 검거 서울 연합뉴스 경찰 관계자들이 19일 오후 서울 강북구 오패산 터널 인근에서 사제 총기를 발사해 경찰을 살해한 용의자 성모씨를 검거하고 있다 성씨는 검거 당시 서바이벌 게임에서 쓰는 방탄조끼에 헬멧까지 착용한 상태였다',
    '서울 연합뉴스 김은경 기자 사제 총기로 경찰을 살해한 범인 성모 46 씨는 주도면밀했다',
    '경찰에 따르면 성씨는 19일 오후 강북경찰서 인근 부동산 업소 밖에서 부동산업자 이모 67 씨가 나오기를 기다렸다 이씨와는 평소에도 말다툼을 자주 한 것으로 알려졌다',
    '이씨가 나와 걷기 시작하자 성씨는 따라가면서 미리 준비해온 사제 총기를 이씨에게 발사했다 총알이 빗나가면서 이씨는 도망갔다 그 빗나간 총알은 지나가던 행인 71 씨의 배를 스쳤다',
    '성씨는 강북서 인근 치킨집까지 이씨 뒤를 쫓으며 실랑이하다 쓰러뜨린 후 총기와 함께 가져온 망치로 이씨 머리를 때렸다',
    '이 과정에서 오후 6시 20분께 강북구 번동 길 위에서 사람들이 싸우고 있다 총소리가 났다 는 등의 신고가 여러건 들어왔다',
    '5분 후에 성씨의 전자발찌가 훼손됐다는 신고가 보호관찰소 시스템을 통해 들어왔다 성범죄자로 전자발찌를 차고 있던 성씨는 부엌칼로 직접 자신의 발찌를 끊었다',
    '용의자 소지 사제총기 2정 서울 연합뉴스 임헌정 기자 서울 시내에서 폭행 용의자가 현장 조사를 벌이던 경찰관에게 사제총기를 발사해 경찰관이 숨졌다 19일 오후 6시28분 강북구 번동에서 둔기로 맞았다 는 폭행 피해 신고가 접수돼 현장에서 조사하던 강북경찰서 번동파출소 소속 김모 54 경위가 폭행 용의자 성모 45 씨가 쏜 사제총기에 맞고 쓰러진 뒤 병원에 옮겨졌으나 숨졌다 사진은 용의자가 소지한 사제총기',
    '신고를 받고 번동파출소에서 김창호 54 경위 등 경찰들이 오후 6시 29분께 현장으로 출동했다 성씨는 그사이 부동산 앞에 놓아뒀던 가방을 챙겨 오패산 쪽으로 도망간 후였다',
    '김 경위는 오패산 터널 입구 오른쪽의 급경사에서 성씨에게 접근하다가 오후 6시 33분께 풀숲에 숨은 성씨가 허공에 난사한 10여발의 총알 중 일부를 왼쪽 어깨 뒷부분에 맞고 쓰러졌다',
    '김 경위는 구급차가 도착했을 때 이미 의식이 없었고 심폐소생술을 하며 병원으로 옮겨졌으나 총알이 폐를 훼손해 오후 7시 40분께 사망했다',
    '김 경위는 외근용 조끼를 입고 있었으나 총알을 막기에는 역부족이었다',
    '머리에 부상을 입은 이씨도 함께 병원으로 이송됐으나 생명에는 지장이 없는 것으로 알려졌다',
    '성씨는 오패산 터널 밑쪽 숲에서 오후 6시 45분께 잡혔다',
    '총격현장 수색하는 경찰들 서울 연합뉴스 이효석 기자 19일 오후 서울 강북구 오패산 터널 인근에서 경찰들이 폭행 용의자가 사제총기를 발사해 경찰관이 사망한 사건을 조사 하고 있다',
    '총 때문에 쫓던 경관들과 민간인들이 몸을 숨겼는데 인근 신발가게 직원 이모씨가 다가가 성씨를 덮쳤고 이어 현장에 있던 다른 상인들과 경찰이 가세해 체포했다',
    '성씨는 경찰에 붙잡힌 직후 나 자살하려고 한 거다 맞아 죽어도 괜찮다 고 말한 것으로 전해졌다',
    '성씨 자신도 경찰이 발사한 공포탄 1발 실탄 3발 중 실탄 1발을 배에 맞았으나 방탄조끼를 입은 상태여서 부상하지는 않았다',
    '경찰은 인근을 수색해 성씨가 만든 사제총 16정과 칼 7개를 압수했다 실제 폭발할지는 알 수 없는 요구르트병에 무언가를 채워두고 심지를 꽂은 사제 폭탄도 발견됐다',
    '일부는 숲에서 발견됐고 일부는 성씨가 소지한 가방 안에 있었다'
]

# In[40]:


# 텍스트랭크 알고리즘
# 0
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
import math
import scipy as sp
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

# In[41]:


# rank 1
import numpy as np
from sklearn.preprocessing import normalize


def pagerank(x, df=0.85, max_iter=30, bias=None):
    """
    Arguments
    ---------
    x : scipy.sparse.csr_matrix
        shape = (n vertex, n vertex)
    df : float
        Damping factor, 0 < df < 1
    max_iter : int
        Maximum number of iteration
    bias : numpy.ndarray or None
        If None, equal bias

    Returns
    -------
    R : numpy.ndarray
        PageRank vector. shape = (n vertex, 1)
    """

    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1, 1)

    # check bias
    if bias is None:
        bias = (1 - df) * np.ones(A.shape[0]).reshape(-1, 1)
    else:
        bias = bias.reshape(-1, 1)
        bias = A.shape[0] * bias / bias.sum()
        assert bias.shape[0] == A.shape[0]
        bias = (1 - df) * bias

    # iteration
    for _ in range(max_iter):
        R = df * (A * R) + bias

    return R


# In[42]:


# utils 2
from collections import Counter
from scipy.sparse import csr_matrix


def scan_vocabulary(sents, tokenize=None, min_count=2):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        tokenize(str) returns list of str
    min_count : int
        Minumum term frequency

    Returns
    -------
    idx_to_vocab : list of str
        Vocabulary list
    vocab_to_idx : dict
        Vocabulary to index mapper.
    """
    counter = Counter(w for sent in sents for w in tokenize(sent))
    counter = {w: c for w, c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x: -x[1])]
    vocab_to_idx = {vocab: idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx


def tokenize_sents(sents, tokenize):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        tokenize(sent) returns list of str (word sequence)

    Returns
    -------
    tokenized sentence list : list of list of str
    """
    return [tokenize(sent) for sent in sents]


def vectorize(tokens, vocab_to_idx):
    """
    Arguments
    ---------
    tokens : list of list of str
        Tokenzed sentence list
    vocab_to_idx : dict
        Vocabulary to index mapper

    Returns
    -------
    sentence bow : scipy.sparse.csr_matrix
        shape = (n_sents, n_terms)
    """
    rows, cols, data = [], [], []
    for i, tokens_i in enumerate(tokens):
        for t, c in Counter(tokens_i).items():
            j = vocab_to_idx.get(t, -1)
            if j == -1:
                continue
            rows.append(i)
            cols.append(j)
            data.append(c)
    n_sents = len(tokens)
    n_terms = len(vocab_to_idx)
    x = csr_matrix((data, (rows, cols)), shape=(n_sents, n_terms))
    return x


# In[43]:


# word 3

def word_graph(sents, tokenize=None, min_count=2, window=2,
               min_cooccurrence=2, vocab_to_idx=None, verbose=False):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        tokenize(str) returns list of str
    min_count : int
        Minumum term frequency
    window : int
        Co-occurrence window size
    min_cooccurrence : int
        Minimum cooccurrence frequency
    vocab_to_idx : dict
        Vocabulary to index mapper.
        If None, this function scan vocabulary first.
    verbose : Boolean
        If True, verbose mode on

    Returns
    -------
    co-occurrence word graph : scipy.sparse.csr_matrix
    idx_to_vocab : list of str
        Word list corresponding row and column
    """
    if vocab_to_idx is None:
        idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    else:
        idx_to_vocab = [vocab for vocab, _ in sorted(vocab_to_idx.items(), key=lambda x: x[1])]

    tokens = tokenize_sents(sents, tokenize)
    g = cooccurrence(tokens, vocab_to_idx, window, min_cooccurrence, verbose)
    return g, idx_to_vocab


def cooccurrence(tokens, vocab_to_idx, window=2, min_cooccurrence=2, verbose=False):
    """
    Arguments
    ---------
    tokens : list of list of str
        Tokenized sentence list
    vocab_to_idx : dict
        Vocabulary to index mapper
    window : int
        Co-occurrence window size
    min_cooccurrence : int
        Minimum cooccurrence frequency
    verbose : Boolean
        If True, verbose mode on

    Returns
    -------
    co-occurrence matrix : scipy.sparse.csr_matrix
        shape = (n_vocabs, n_vocabs)
    """
    counter = defaultdict(int)
    for s, tokens_i in enumerate(tokens):
        if verbose and s % 1000 == 0:
            print('\rword cooccurrence counting {}'.format(s), end='')
        vocabs = [vocab_to_idx[w] for w in tokens_i if w in vocab_to_idx]
        n = len(vocabs)
        for i, v in enumerate(vocabs):
            if window <= 0:
                b, e = 0, n
            else:
                b = max(0, i - window)
                e = min(i + window, n)
            for j in range(b, e):
                if i == j:
                    continue
                counter[(v, vocabs[j])] += 1
                counter[(vocabs[j], v)] += 1
    counter = {k: v for k, v in counter.items() if v >= min_cooccurrence}
    n_vocabs = len(vocab_to_idx)
    if verbose:
        print('\rword cooccurrence counting from {} sents was done'.format(s + 1))
    return dict_to_mat(counter, n_vocabs, n_vocabs)


def dict_to_mat(d, n_rows, n_cols):
    """
    Arguments
    ---------
    d : dict
        key : (i,j) tuple
        value : float value

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    rows, cols, data = [], [], []
    for (i, j), v in d.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


# In[44]:


# sentence 4

from collections import Counter
import math
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances


def sent_graph(sents, tokenize=None, min_count=2, min_sim=0.3,
               similarity=None, vocab_to_idx=None, verbose=False):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        tokenize(sent) return list of str
    min_count : int
        Minimum term frequency
    min_sim : float
        Minimum similarity between sentences
    similarity : callable or str
        similarity(s1, s2) returns float
        s1 and s2 are list of str.
        available similarity = [callable, 'cosine', 'textrank']
    vocab_to_idx : dict
        Vocabulary to index mapper.
        If None, this function scan vocabulary first.
    verbose : Boolean
        If True, verbose mode on

    Returns
    -------
    sentence similarity graph : scipy.sparse.csr_matrix
        shape = (n sents, n sents)
    """

    if vocab_to_idx is None:
        idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    else:
        idx_to_vocab = [vocab for vocab, _ in sorted(vocab_to_idx.items(), key=lambda x: x[1])]

    x = vectorize_sents(sents, tokenize, vocab_to_idx)
    if similarity == 'cosine':
        x = numpy_cosine_similarity_matrix(x, min_sim, verbose, batch_size=1000)
    else:
        x = numpy_textrank_similarity_matrix(x, min_sim, verbose, batch_size=1000)
    return x


def vectorize_sents(sents, tokenize, vocab_to_idx):
    rows, cols, data = [], [], []
    for i, sent in enumerate(sents):
        counter = Counter(tokenize(sent))
        for token, count in counter.items():
            j = vocab_to_idx.get(token, -1)
            if j == -1:
                continue
            rows.append(i)
            cols.append(j)
            data.append(count)
    n_rows = len(sents)
    n_cols = len(vocab_to_idx)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def numpy_cosine_similarity_matrix(x, min_sim=0.3, verbose=True, batch_size=1000):
    n_rows = x.shape[0]
    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx + 1) * batch_size))
        psim = 1 - pairwise_distances(x[b:e], x, metric='cosine')
        rows, cols = np.where(psim >= min_sim)
        data = psim[rows, cols]
        mat.append(csr_matrix((data, (rows, cols)), shape=(e - b, n_rows)))
        if verbose:
            print('\rcalculating cosine sentence similarity {} / {}'.format(b, n_rows), end='')
    mat = sp.sparse.vstack(mat)
    if verbose:
        print('\rcalculating cosine sentence similarity was done with {} sents'.format(n_rows))
    return mat


def numpy_textrank_similarity_matrix(x, min_sim=0.3, verbose=True, min_length=1, batch_size=1000):
    n_rows, n_cols = x.shape

    # Boolean matrix
    rows, cols = x.nonzero()
    data = np.ones(rows.shape[0])
    z = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Inverse sentence length
    size = np.asarray(x.sum(axis=1)).reshape(-1)
    size[np.where(size <= min_length)] = 10000
    size = np.log(size)

    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):

        # slicing
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx + 1) * batch_size))

        # dot product
        inner = z[b:e, :] * z.transpose()

        # sentence len[i,j] = size[i] + size[j]
        norm = size[b:e].reshape(-1, 1) + size.reshape(1, -1)
        norm = norm ** (-1)
        norm[np.where(norm == np.inf)] = 0

        # normalize
        sim = inner.multiply(norm).tocsr()
        rows, cols = (sim >= min_sim).nonzero()
        data = np.asarray(sim[rows, cols]).reshape(-1)

        # append
        mat.append(csr_matrix((data, (rows, cols)), shape=(e - b, n_rows)))

        if verbose:
            print('\rcalculating textrank sentence similarity {} / {}'.format(b, n_rows), end='')

    mat = sp.sparse.vstack(mat)
    if verbose:
        print('\rcalculating textrank sentence similarity was done with {} sents'.format(n_rows))

    return mat


def graph_with_python_sim(tokens, verbose, similarity, min_sim):
    if similarity == 'cosine':
        similarity = cosine_sent_sim
    elif callable(similarity):
        similarity = similarity
    else:
        similarity = textrank_sent_sim

    rows, cols, data = [], [], []
    n_sents = len(tokens)
    for i, tokens_i in enumerate(tokens):
        if verbose and i % 1000 == 0:
            print('\rconstructing sentence graph {} / {} ...'.format(i, n_sents), end='')
        for j, tokens_j in enumerate(tokens):
            if i >= j:
                continue
            sim = similarity(tokens_i, tokens_j)
            if sim < min_sim:
                continue
            rows.append(i)
            cols.append(j)
            data.append(sim)
    if verbose:
        print('\rconstructing sentence graph was constructed from {} sents'.format(n_sents))
    return csr_matrix((data, (rows, cols)), shape=(n_sents, n_sents))


def textrank_sent_sim(s1, s2):
    """
    Arguments
    ---------
    s1, s2 : list of str
        Tokenized sentences

    Returns
    -------
    Sentence similarity : float
        Non-negative number
    """
    n1 = len(s1)
    n2 = len(s2)
    if (n1 <= 1) or (n2 <= 1):
        return 0
    common = len(set(s1).intersection(set(s2)))
    base = math.log(n1) + math.log(n2)
    return common / base


def cosine_sent_sim(s1, s2):
    """
    Arguments
    ---------
    s1, s2 : list of str
        Tokenized sentences

    Returns
    -------
    Sentence similarity : float
        Non-negative number
    """
    if (not s1) or (not s2):
        return 0

    s1 = Counter(s1)
    s2 = Counter(s2)
    norm1 = math.sqrt(sum(v ** 2 for v in s1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in s2.values()))
    prod = 0
    for k, v in s1.items():
        prod += v * s2.get(k, 0)
    return prod / (norm1 * norm2)


# In[45]:


# summarizer 5

class KeywordSummarizer:
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        Tokenize function: tokenize(str) = list of str
    min_count : int
        Minumum frequency of words will be used to construct sentence graph
    window : int
        Word cooccurrence window size. Default is -1.
        '-1' means there is cooccurrence between two words if the words occur in a sentence
    min_cooccurrence : int
        Minimum cooccurrence frequency of two words
    vocab_to_idx : dict or None
        Vocabulary to index mapper
    df : float
        PageRank damping factor
    max_iter : int
        Number of PageRank iterations
    verbose : Boolean
        If True, it shows training progress
    """

    def __init__(self, sents=None, tokenize=None, min_count=2,
                 window=-1, min_cooccurrence=2, vocab_to_idx=None,
                 df=0.85, max_iter=30, verbose=False):

        self.tokenize = tokenize
        self.min_count = min_count
        self.window = window
        self.min_cooccurrence = min_cooccurrence
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter
        self.verbose = verbose

        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents, bias=None):
        """
        Arguments
        ---------
        sents : list of str
            Sentence list
        bias : None or numpy.ndarray
            PageRank bias term

        Returns
        -------
        None
        """

        g, self.idx_to_vocab = word_graph(sents,
                                          self.tokenize, self.min_count, self.window,
                                          self.min_cooccurrence, self.vocab_to_idx, self.verbose)
        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
        if self.verbose:
            print('trained TextRank. n words = {}'.format(self.R.shape[0]))

    def keywords(self, topk=30):
        """
        Arguments
        ---------
        topk : int
            Number of keywords selected from TextRank

        Returns
        -------
        keywords : list of tuple
            Each tuple stands for (word, rank)
        """
        if not hasattr(self, 'R'):
            raise RuntimeError('Train textrank first or use summarize function')
        idxs = self.R.argsort()[-topk:]
        keywords = [(self.idx_to_vocab[idx], self.R[idx]) for idx in reversed(idxs)]
        return keywords

    def summarize(self, sents, topk=30):
        """
        Arguments
        ---------
        sents : list of str
            Sentence list
        topk : int
            Number of keywords selected from TextRank

        Returns
        -------
        keywords : list of tuple
            Each tuple stands for (word, rank)
        """
        self.train_textrank(sents)
        return self.keywords(topk)


class KeysentenceSummarizer:
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        Tokenize function: tokenize(str) = list of str
    min_count : int
        Minumum frequency of words will be used to construct sentence graph
    min_sim : float
        Minimum similarity between sentences in sentence graph
    similarity : str
        available similarity = ['cosine', 'textrank']
    vocab_to_idx : dict or None
        Vocabulary to index mapper
    df : float
        PageRank damping factor
    max_iter : int
        Number of PageRank iterations
    verbose : Boolean
        If True, it shows training progress
    """

    def __init__(self, sents=None, tokenize=None, min_count=2,
                 min_sim=0.3, similarity=None, vocab_to_idx=None,
                 df=0.85, max_iter=30, verbose=False):

        self.tokenize = tokenize
        self.min_count = min_count
        self.min_sim = min_sim
        self.similarity = similarity
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter
        self.verbose = verbose

        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents, bias=None):
        """
        Arguments
        ---------
        sents : list of str
            Sentence list
        bias : None or numpy.ndarray
            PageRank bias term
            Shape must be (n_sents,)

        Returns
        -------
        None
        """
        g = sent_graph(sents, self.tokenize, self.min_count,
                       self.min_sim, self.similarity, self.vocab_to_idx, self.verbose)
        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
        if self.verbose:
            print('trained TextRank. n sentences = {}'.format(self.R.shape[0]))

    def summarize(self, sents, topk=30, bias=None):
        """
        Arguments
        ---------
        sents : list of str
            Sentence list
        topk : int
            Number of key-sentences to be selected.
        bias : None or numpy.ndarray
            PageRank bias term
            Shape must be (n_sents,)

        Returns
        -------
        keysents : list of tuple
            Each tuple stands for (sentence index, rank, sentence)

        Usage
        -----
            >>> from textrank import KeysentenceSummarizer

            >>> summarizer = KeysentenceSummarizer(tokenize = tokenizer, min_sim = 0.5)
            >>> keysents = summarizer.summarize(texts, topk=30)
        """
        n_sents = len(sents)
        if isinstance(bias, np.ndarray):
            if bias.shape != (n_sents,):
                raise ValueError('The shape of bias must be (n_sents,) but {}'.format(bias.shape))
        elif bias is not None:
            raise ValueError('The type of bias must be None or numpy.ndarray but the type is {}'.format(type(bias)))

        self.train_textrank(sents, bias)
        idxs = self.R.argsort()[-topk:]
        keysents = [(idx, self.R[idx], sents[idx]) for idx in reversed(idxs)]
        return keysents

#//////////////////////////////////////////////////////////////////////////////////////////////////////////

from konlpy.tag import Okt
from konlpy.tag import Komoran
from konlpy.tag import Kkma
from textrank import KeysentenceSummarizer
from textrank import KeywordSummarizer
import numpy as np


okt = Okt()
komoran = Komoran()
kkma = Kkma()

def okt_tokenize(sent):
    words = okt.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV')]
    return words

# def komoran_tokenize(sent):
#     words = komoran.pos(sent, join=True)
#     words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV')]
#     return words
#
# def kkma_tokenize(sent):
#     words = kkma.pos(sent, join=True)
#     words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV')]
#     return words

def subword_tokenizer(sent, n=10):
    def subword(token, n):
        if len(token) <= n:
            return [token]
        return [token[i:i+n] for i in range(len(token) - n)]
    return [sub for token in sent.split() for sub in subword(token, n)]

def okt_tokenizer(sents):
    return okt.nouns(sents)


def okt_nouns_ox(sents):
    word_counts = {}

    for sent in sents:
        nouns = okt.nouns(sent)
        for noun in nouns:
            if len(noun) == 1:
                continue
            if noun not in word_counts:
                word_counts[noun] = 0
            word_counts[noun] += 1

    return word_counts

def okt_tokenizer_ox(sents):
    return okt_nouns_ox(okt.nouns(sents))

# 키워드 뽑기
for i in range(1, 158):
    f = open("C:/Users/SAMSUNG/PycharmProjects/Web_Crawling/" + str(i) + ".txt", 'rt', encoding='UTF-8')
    lines = f.read().splitlines()
    lines = list(filter(None, lines))
    data = ' '
    for line in lines:
        if line == lines[0]:
            continue
        elif line == lines[1]:
            continue
        else:
            line = line.strip()
            data = data + line
            data = data + "\n"

    # 구 단위로
    sents = okt.phrases(data)
    summarizer = KeywordSummarizer(tokenize=okt_tokenizer_ox, min_count=0, min_cooccurrence=1)
    keywords1 = summarizer.summarize(sents, topk=30)

    # 띄어쓰기로 나눈거
    sents = subword_tokenizer(data)
    summarizer = KeywordSummarizer(tokenize=okt_tokenizer_ox, min_count=0, min_cooccurrence=1)
    keywords2 = summarizer.summarize(sents, topk=30)

    keywords_1 = []
    keywords_2 = []

    for j in range(1, 31):
        keywords_1.append(keywords1[j-1][0])
        keywords_2.append(keywords2[j-1][0])

    f = open("key_word_" + str(i) + ".txt", "w", encoding='UTF-8')

    for l in range(30):
        f.write(keywords_1[l] + ',' + keywords_2[l] + ',')

    f.close()

# 키문장 뽑기
for i in range(1, 158):
    f = open("C:/Users/SAMSUNG/PycharmProjects/Web_Crawling/" + str(i) + ".txt", 'rt', encoding='UTF-8')
    lines = f.read().splitlines()
    lines = list(filter(None, lines))
    sents = []
    for line in lines:
        line = line.strip()
        sents.append(line)

    del sents[0]
    del sents[0]

    summarizer = KeysentenceSummarizer(tokenize=okt_tokenize, min_sim=0.5)
    keysents = summarizer.summarize(sents, topk=2)

    f = open('key_sent_' + str(i) + '.txt', 'w', -1, 'utf-8')
    f.write(keysents[0][2])
    f.write("\n")
    f.write(keysents[1][2])
    f.close()