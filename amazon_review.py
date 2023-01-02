"""
Rating Product & Sorting Reviews
"""

# Business Problem
"""
One of the most important problems in e-commerce is accurately calculating the ratings given to products after the sale.
Solving this problem means providing more customer satisfaction for the e-commerce website, highlighting the product for the seller, 
and providing a hassle-free shopping experience for the buyer. Another problem is correctly sorting the reviews given for products.
Misleading reviews that stand out can directly affect the sales of the product, causing both financial loss and customer loss.
Solving these two fundamental problems will increase sales for the e-commerce website and sellers while allowing customers to complete
their shopping journey smoothly.
"""

# Features
"""
- reviewerID --> User Id
- asin --> Product Id
- reviewerName --> User name 
- helpful --> Useful Evaluation Degree
- reviewText --> Evaluation
- overall --> Product Rating
- summary --> Evaluation Summary
- unixReviewTime --> Evaluation Time
- reviewTime --> Evaluation Time {RAW}
- day_diff --> Number of days since assessment
- helpful_yes --> The number of times the evaluation was found useful
- total_vote --> Number of votes given to the evaluation
"""
import pandas as pd
import math
import scipy.stats as st

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_csv("amazon_review.csv")
df.head()

df.shape  # (4915, 12)

df.info()
"""
 0   reviewerID      4915 non-null   object 
 1   asin            4915 non-null   object 
 2   reviewerName    4914 non-null   object 
 3   helpful         4915 non-null   object 
 4   reviewText      4914 non-null   object 
 5   overall         4915 non-null   float64
 6   summary         4915 non-null   object 
 7   unixReviewTime  4915 non-null   int64  
 8   reviewTime      4915 non-null   object 
 9   day_diff        4915 non-null   int64  
 10  helpful_yes     4915 non-null   int64  
 11  total_vote      4915 non-null   int64  

"""
df.loc[df["reviewText"].isnull()]

df.describe().T
"""
                    count             mean            std              min              25%              50%              75%              max
overall        4915.00000          4.58759        0.99685          1.00000          5.00000          5.00000          5.00000          5.00000
unixReviewTime 4915.00000 1379465001.66836 15818574.32275 1339200000.00000 1365897600.00000 1381276800.00000 1392163200.00000 1406073600.00000
day_diff       4915.00000        437.36704      209.43987          1.00000        281.00000        431.00000        601.00000       1064.00000
helpful_yes    4915.00000          1.31109       41.61916          0.00000          0.00000          0.00000          0.00000       1952.00000
total_vote     4915.00000          1.52146       44.12309          0.00000          0.00000          0.00000          0.00000       2020.00000
"""
df.groupby("asin").agg({"overall": "mean"})  # 4.58759

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()  # Timestamp('2014-12-07 00:00:00')
df["days"] = (current_date - df["reviewTime"]).dt.days

df["days"].describe([.1, .25, .5, .7, .8, .9, .95, .99]).T
"""
count   4915.00000
mean     436.36704
std      209.43987
min        0.00000
10%      166.00000
25%      280.00000
50%      430.00000
70%      561.80000
80%      637.00000
90%      707.00000
95%      747.00000
99%      942.00000
max     1063.00000
"""
df["days_cat"] = pd.qcut(df["days"], q=4, labels=["q1", "q2", "q3", "q4"])
df.groupby("days_cat").agg({"overall": "mean"})
"""
q1        4.69579
q2        4.63614
q3        4.57166
q4        4.44625
"""
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df = df[["overall", "reviewTime", "day_diff", "helpful_yes", "helpful_no", "total_vote"]]
df.sample(5)
"""
      overall reviewTime  day_diff  helpful_yes  helpful_no  total_vote  
3111  1.00000 2014-09-07        92            1           0           1            
4070  2.00000 2014-06-02       189            0           0           0                
3345  5.00000 2014-06-01       190            2           0           2                       
970   5.00000 2013-02-07       669            0           0           0                     
53    4.00000 2013-10-07       427            0           0           0        
"""
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head()

df.sort_values("wilson_lower_bound", ascending=False).head(20)
"""
      overall reviewTime  day_diff  helpful_yes  helpful_no  total_vote  score_pos_neg_diff  score_average_rating  wilson_lower_bound
2031  5.00000 2013-01-05       702         1952          68        2020                1884               0.96634             0.95754
3449  5.00000 2012-09-26       803         1428          77        1505                1351               0.94884             0.93652
4212  1.00000 2013-05-08       579         1568         126        1694                1442               0.92562             0.91214
317   1.00000 2012-02-09      1033          422          73         495                 349               0.85253             0.81858
4672  5.00000 2014-07-03       158           45           4          49                  41               0.91837             0.80811
1835  5.00000 2014-02-28       283           60           8          68                  52               0.88235             0.78465
3981  5.00000 2012-10-22       777          112          27         139                  85               0.80576             0.73214
3807  3.00000 2013-02-27       649           22           3          25                  19               0.88000             0.70044
4306  5.00000 2012-09-06       823           51          14          65                  37               0.78462             0.67033
4596  1.00000 2012-09-22       807           82          27         109                  55               0.75229             0.66359
315   5.00000 2012-08-13       847           38          10          48                  28               0.79167             0.65741
1465  4.00000 2014-04-14       238            7           0           7                   7               1.00000             0.64567
1609  5.00000 2014-03-26       257            7           0           7                   7               1.00000             0.64567
4302  5.00000 2014-03-21       262           14           2          16                  12               0.87500             0.63977
4072  5.00000 2012-11-09       759            6           0           6                   6               1.00000             0.60967
1072  5.00000 2012-05-10       942            5           0           5                   5               1.00000             0.56552
2583  5.00000 2013-08-06       489            5           0           5                   5               1.00000             0.56552
121   5.00000 2012-05-09       943            5           0           5                   5               1.00000             0.56552
1142  5.00000 2014-02-04       307            5           0           5                   5               1.00000             0.56552
1753  5.00000 2012-10-22       777            5           0           5                   5               1.00000             0.56552

"""
