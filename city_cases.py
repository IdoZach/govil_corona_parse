import pandas as pd
import matplotlib.pyplot as plt
import datetime, os
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, LogisticRegression
import plotly.graph_objects as go

def get_population():
    # src: wikipedia, march 29 2020: https://he.wikipedia.org/wiki/%D7%AA%D7%91%D7%A0%D7%99%D7%AA:%D7%94%D7%A2%D7%A8%D7%99%D7%9D_%D7%94%D7%92%D7%93%D7%95%D7%9C%D7%95%D7%AA_%D7%A9%D7%9C_%D7%99%D7%A9%D7%A8%D7%90%D7%9C
    cities="""1 	ירושלים 	ירושלים 	865,700 	11 	רמת גן 	תל אביב 	152,600
    2 	תל אביב 	תל אביב 	432,900 	12 	רחובות 	המרכז 	132,700
    3 	חיפה 	חיפה 	278,900 	13 	אשקלון 	הדרום 	130,700
    4 	ראשון לציון 	המרכז 	247,323 	14 	בת ים 	תל אביב 	128,900
    5 	פתח תקווה 	המרכז 	231,000 	15 	בית שמש 	ירושלים 	103,900
    6 	אשדוד 	הדרום 	220,200 	16 	כפר סבא 	המרכז 	96,900
    7 	נתניה 	המרכז 	207,900 	17 	הרצליה 	תל אביב 	91,900
    8 	באר שבע 	הדרום 	203,600 	18 	חדרה 	חיפה 	88,800
    9 	חולון 	תל אביב 	188,800 	19 	מודיעין-מכבים-רעות 	המרכז 	88,700
    10 	בני ברק 	תל אביב 	182,800 	20 	נצרת 	הצפון 	75,700 """
    cities = np.array([x.strip() for x in '\t'.join(cities.split('\n')).split('\t')])
    cities = cities.reshape(-1,4)
    cities = cities[:,[1,3]]
    df = pd.DataFrame(cities)
    df.columns=['city','pop']
    df['pop'] = df['pop'].apply(lambda x: int(x.replace(',','')))
    return df

def input_cases_per_city(cache = 'inputted_from_walla_3349123.csv'):
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        df = get_population()
        confirmed = []
        general = pd.DataFrame([['All',8546000]]) # population as of 2016, mattching the wikipedia city population year.
        general.columns=['city','pop']
        df = df.append(general)
        # filled via https://news.walla.co.il/item/3349123
        for i,(k,row) in enumerate(df.iterrows()):
            print('{}/{} enter number of confirmed for city/region {}'.format(i,len(df),row['city']))
            conf = input()
            conf = 0 if conf == '' else conf
            confirmed.append(int(conf))
        df['confirmed'] = confirmed
        print(df)
        df.to_csv(cache,index=None)
    return df

def plot_pie_chart_px(df):
    tot_pop = df.query('city == "All"')['pop']
    df = df.query('city != "All"')
    radius = df['ratio']
    # theta goes from 0 to 360-1
    width = np.array(df['pop'] / float(tot_pop) * 360)
    theta = np.cumsum(np.concatenate((width,[0])))[:-1] - width/2
    labels = np.array(df['city'])
    fig = go.Figure(go.Barpolar(
        r=radius,
        theta=theta,
        width=width,
        marker_color=3*["#E4FF87", '#709BFF', '#709BFF', '#FFAA70', '#FFAA70', '#FFDF70', '#B6FFB4'],
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.8
    ))
    r = np.linspace(0,1,10)
    fig.update_layout(
        title="Confirmed cases (% per city), March 27th",
        template=None,
        polar_angularaxis = dict(tickvals=theta,
                             ticktext=labels),
        polar_radialaxis = dict(tickvals=r,
                             ticktext=['{:1.2f}%'.format(x) for x in r])
    )
    fig.show()
if __name__ == '__main__':
    df = input_cases_per_city()

    df = df.query('confirmed > 0')
    df['ratio'] = np.round(100* 1.*df['confirmed'] / df['pop'],3)
    df['city_rev'] = df['city'].apply(lambda x: x[::-1] if x != 'All' else x)
    plot_pie_chart_px(df)

    # simple bar:
    #ax = df.plot.bar(x='city_rev',y='ratio',rot=45)
    #ax.set_ylabel('%')
    #plt.show()

