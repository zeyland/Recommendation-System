
# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


df_ = pd.read_excel("Dataset/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()


def outlier_thresholds(dataframe, variable): #eşit değerlerini belirle
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable): #aykırı değerleri baskıla
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.columns
df.head()


df_de = df[df['Country'] == "Germany"]



# Biz ürünler sütunda olsun istiyoruz. unstack fonksiyonu.
# kaç tane olduğuyla ilgilenmiyoruz sadece var ve yok.
# apply map tüm satır sütunlarda gezmek için kullandığımız fonk. tüm hücreler

def create_invoice_product_df(dataframe, id=False):
    if id: #id girildiğinde döner. ama öntanımlı olarak else kısmı çalışır. stock koduna göre döndü.
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
 #id = true ise stock koduna göre döndürür.
de_inv_pro_df = create_invoice_product_df(df_de, id=True)
de_inv_pro_df.head()




de_inv_pro_df = create_invoice_product_df(df_de)
de_inv_pro_df.head()
de_inv_pro_df.info()
#[457 rows x 1696 columns]
#Invoice 457 x Description 1696


de_inv_pro_df = create_invoice_product_df(df_de, id=True)
df_de.info()
df_de.columns
#[457 rows x 1664 columns]
#Invoice 457 x StockCode 1664

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)
df_de.nunique()
# Görev 3:
# Kullanıcı 1 ürün id'si: 21987
# Kullanıcı 2 ürün id'si: 23235
# Kullanıcı 3 ürün id'si: 22747


check_id(df_de, 21987)
check_id(df_de, 23235)
check_id(df_de, 22747)

# ['PACK OF 6 SKULL PAPER CUPS']
# ['STORAGE TIN VINTAGE LEAF']
# ["POPPY'S PLAYHOUSE BATHROOM"]
###############################################

# min support  0.01 demek ürünün gözükme olasılığı bundan düşükse getirme hiç veriyi demek. 1/2/3 lü hepsini.
# apiriori= her bir ürünle diğer ürünlerle gözükme olasılığı
frequent_itemsets = apriori(de_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head()

# birliktelik kuralı -confidence support ünemli.
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()
rules.sort_values("lift", ascending=False).head(500)
rules.head()
############################################
# Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################
#laptop alana çantayı önerme gibi.


# Kullanıcı 1 ürün id'si: 21987
# Kullanıcı 2 ürün id'si: 23235
# Kullanıcı 3 ürün id'si: 22747
#KULLANICILARLA İLGİLENMİYORUM ODAĞIM ÜRÜN.

product_id = 21987
check_id(df, product_id)


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]


arl_recommender(rules, 21987, 1)
check_id(df, 21244)
#21244 ['BLUE POLKADOT PLATE ']

arl_recommender(rules, 23235, 1)
check_id(df, 21244)


arl_recommender(rules, 22747, 1)
df_de[df_de["Customer ID"]== check_id(df, 20750)]
#20750 ['RED RETROSPOT MINI CASES']



