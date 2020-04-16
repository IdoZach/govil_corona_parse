import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np, re
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, LogisticRegression
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MinuteLocator
import matplotlib.gridspec as gridspec
import matplotlib._color_data as mcd
def corona_israel_city():
    df = pd.DataFrame([
        # this is always in the morning
        dict(Date='3/31/2020', jerusalem=650, bnei_brak=571, ashkelon=114, haifa=67, tel_aviv=278, kiryat_yam=13),
        dict(Date='4/1/2020', jerusalem=807, bnei_brak=723, ashkelon=125, haifa=81, tel_aviv=301, kiryat_yam=16), # kiryat_yam - estimate
        dict(Date='4/3/2020', jerusalem=1132, bnei_brak=1061, ashkelon=170, haifa=96, tel_aviv=337, kiryat_yam=19),
        dict(Date='4/5/2020', jerusalem=1302, bnei_brak=1214, ashkelon=191, haifa=105, tel_aviv=359, kiryat_yam=20),
        dict(Date='4/6/2020', jerusalem=1424, bnei_brak=1323, ashkelon=207, haifa=106, tel_aviv=387, kiryat_yam=21),
        dict(Date='4/7/2020', jerusalem=1464, bnei_brak=1386, ashkelon=209, haifa=108, tel_aviv=393, kiryat_yam=21),
        dict(Date='4/9/2020', jerusalem=1630, bnei_brak=1594, ashkelon=216, haifa=118, tel_aviv=415, kiryat_yam=21),
        dict(Date='4/10/2020', jerusalem=1780, bnei_brak=1681, ashkelon=220, haifa=122, tel_aviv=434, kiryat_yam=21), # haifa is just an apprx here, not reported.
        dict(Date='4/11/2020', jerusalem=1821, bnei_brak=1761, ashkelon=225, haifa=127, tel_aviv=444, kiryat_yam=21),
        dict(Date='4/12/2020', jerusalem=1959, bnei_brak=1806, ashkelon=228, haifa=134, tel_aviv=452, kiryat_yam=21),
        dict(Date='4/13/2020', jerusalem=2093, bnei_brak=1888, ashkelon=228, haifa=136, tel_aviv=458, kiryat_yam=21),
        dict(Date='4/14/2020', jerusalem=2258, bnei_brak=2053, ashkelon=235, haifa=139, tel_aviv=468, kiryat_yam=21),


    ])
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df_city_confirmed = corona_israel_city()

def tests_per_city():

    df = pd.DataFrame([
        dict(city='jerusalem', confirmed=1959, tests=18312),
        dict(city='haifa', confirmed=134, tests=3995),
        dict(city='bnei_brak', confirmed=1806, tests=8531),
        dict(city='ashkelon', confirmed=228, tests=3440),
        dict(city='tel_aviv', confirmed=452, tests=11013)
    ])
    df['ratio'] = df['confirmed']/df['tests']
    print('mean ratio in selected cities: {}'.format(df['ratio'].mean()))
    print(df)
    df.plot()
    plt.show()


def get_haifa_confirmed_rank():
    df = pd.DataFrame([
        # this is always in the morning
        dict(Date='3/31/2020', rank=16),
        dict(Date='4/1/2020', rank=15),
        dict(Date='4/3/2020', rank=15),
        dict(Date='4/5/2020', rank=15),
        dict(Date='4/6/2020', rank=15),
        dict(Date='4/7/2020', rank=16),
        dict(Date='4/9/2020', rank=16),
        # dict(Date='4/10/2020',  rank=1),
        dict(Date='4/11/2020', rank=18),

    ])
    df['Date'] = pd.to_datetime(df['Date'])
    df = pd.merge(df, df_city_confirmed, on='Date', how='outer')
    df = df.sort_values('Date').dropna()
    print(df)
    fig,ax=plt.subplots()
    ax1 = ax.twinx()
    col = 'tab:red','tab:blue'
    ax.set_ylabel('City rank',color=col[0])
    ax.invert_yaxis()
    ax1.set_ylabel('Confirmed',color=col[1])
    ax.plot(df['Date'],df['rank'],'.:',color=col[0])
    ax1.plot(df['Date'],df['haifa'],'.-',color=col[1])
    ax.xaxis.set_major_formatter(DateFormatter('%d-%m'))
    ax1.yaxis.set_tick_params(colors=col[1])
    ax.yaxis.set_tick_params(colors=col[0])
    ranks =list(range(int(df['rank'].min()),int(df['rank'].max()+1)))
    ax.yaxis.set_ticks(ranks)
    ax.set_title('Haifa confirmed and rank')
    fig.autofmt_xdate()
    plt.show()

    #return df

def get_deceased_stats():
    df = pd.read_csv('/home/ido/Downloads/Telegram Desktop/deceased_1104.csv')
    df = df.dropna()
    df.columns = ['idx','age','sex','hospital','city']
    #print(df.to_string())
    fig,ax=plt.subplots()
    df['city_he'] = df['city'].apply(lambda x: x[::-1] if type(x) is str else x)
    cities = df['city_he'].unique()
    # filter out cities with 1 deceased only
    good_cities = []
    for city in cities:
        if df.loc[df['city_he'] == city].shape[0] > 1:
            good_cities.append(city)
    #df = df.loc[df['city_he'].isin(good_cities)]
    cities = df['city_he'].unique()
    counts={}
    df['age'] = df['age'].apply(lambda x: np.round(x).astype(int))
    st = 5
    bins=list(range(20,105,st))
    centers = [x+st/2 for x in bins[:-1]]
    for i,city in enumerate(cities):
        c,_ = np.histogram(df.loc[df['city_he']==city,'age'], bins=bins)
        counts[city] = c
    dfc = pd.DataFrame(counts).T
    dfc = dfc.fillna(0)
    colors = mcd.XKCD_COLORS
    leg=[]
    prev=None
    dfc['sum'] = -dfc.sum(axis=1)
    dfc = dfc.sort_values(by='sum').drop('sum',axis=1)

    for i,(k,row) in enumerate(dfc.iterrows()):
        ax.bar(centers,row,bottom=prev,color=colors[list(colors.keys())[5*i]],width=st-1)
        prev=prev+row if prev is not None else row
        leg.append('{}: {}'.format(k,int(df.loc[df['city_he']==k].shape[0])))
    ax.set_yticks(range(0,15,1))
    ax.legend(leg)
    ax.set_xlabel('age')
    ax.set_ylabel('counts')
    ax.set_title('Deceased by age and city (with over 1 deceased), {}'.format(datetime.datetime.now().strftime('%d/%m')))

    plt.show()


def manual_corona_israel():
    # parse corona data manually from IMOH's telegram account (some are just assumptions).

    li = [dict(Date='3/1/2020',confirmed=12,     recovered=0, ventilated=1,deceased=0),
          dict(Date='3/2/2020',confirmed=12,     recovered=0, ventilated=1,deceased=0),
          dict(Date='3/3/2020', confirmed=15,    recovered=0, ventilated=1, deceased=0),
          dict(Date='3/4/2020', confirmed=16,    recovered=0, ventilated=1, deceased=0),
          dict(Date='3/5/2020', confirmed=20,    recovered=0, ventilated=1, deceased=0),
          dict(Date='3/6/2020', confirmed=22,    recovered=0, ventilated=1, deceased=0),
          dict(Date='3/7/2020', confirmed=37,    recovered=0, ventilated=1, deceased=0),
          dict(Date='3/8/2020', confirmed=39,    recovered=0, ventilated=1, deceased=0),
          dict(Date='3/9/2020', confirmed=59,    recovered=0, ventilated=1, deceased=0),
          dict(Date='3/10/2020', confirmed=77,   recovered=3, ventilated=1, deceased=0),
          dict(Date='3/11/2020', confirmed=99,   recovered=3, ventilated=1, deceased=0),
          dict(Date='3/12/2020', confirmed=130,   recovered=3, ventilated=2, deceased=0),
          dict(Date='3/13/2020', confirmed=164,   recovered=3, ventilated=2, deceased=0),
          dict(Date='3/14/2020', confirmed=200,   recovered=4, ventilated=2, deceased=0),
          dict(Date='3/15/2020', confirmed=253,   recovered=4, ventilated=2, deceased=0),
          dict(Date='3/16/2020', confirmed=318,   recovered=4, ventilated=4, deceased=0),
          dict(Date='3/17/2020', confirmed=421,   recovered=11, ventilated=5, deceased=0),
          dict(Date='3/18/2020', confirmed=524,   recovered=11, ventilated=5, deceased=0),
          dict(Date='3/19/2020', confirmed=677,   recovered=14, ventilated=6, deceased=0),
          dict(Date='3/20/2020', confirmed=838,   recovered=15, ventilated=12, deceased=0),
          dict(Date='3/21/2020', confirmed=943,   recovered=37, ventilated=15, deceased=1),
          dict(Date='3/22/2020', confirmed=1207,  recovered=37, ventilated=15, deceased=1),
          dict(Date='3/23/2020', confirmed=1552,  recovered=41, ventilated=29, deceased=1),
          dict(Date='3/24/2020', confirmed=2000,  recovered=53, ventilated=31, deceased=3),
          dict(Date='3/25/2020', confirmed=2463,   recovered=64, ventilated=34, deceased=5),
          dict(Date='3/26/2020', confirmed=2666,   recovered=68, ventilated=37, deceased=8),
          dict(Date='3/27/2020', confirmed=3035,   recovered=79, ventilated=38, deceased=12),
          dict(Date='3/28/2020', confirmed=3618, recovered=89, ventilated=43, deceased=12),
          dict(Date='3/29/2020', confirmed=3944, recovered=132, ventilated=59, deceased=15),
          dict(Date='3/30/2020', confirmed=4695, recovered=161, ventilated=66, deceased=16),
          dict(Date='3/31/2020', confirmed=5358, recovered=224, ventilated=76, deceased=20),
          dict(Date='4/01/2020', confirmed=6092, recovered=241, ventilated=81, deceased=25),
          dict(Date='4/02/2020', confirmed=6857, recovered=338, ventilated=87, deceased=34),
          dict(Date='4/03/2020', confirmed=7428, recovered=403, ventilated=96, deceased=39),
          dict(Date='4/04/2020', confirmed=7851, recovered=458, ventilated=108, deceased=43),
          dict(Date='4/05/2020', confirmed=8430, recovered=546, ventilated=106, deceased=49),
          dict(Date='4/06/2020', confirmed=8904, recovered=670, ventilated=109, deceased=57),
          dict(Date='4/07/2020', confirmed=9248, recovered=770, ventilated=117, deceased=65),
          dict(Date='4/08/2020', confirmed=9755, recovered=864, ventilated=119, deceased=79), # actually 9.4 10am
          dict(Date='4/09/2020', confirmed=9968, recovered=1011, ventilated=121, deceased=86),
          dict(Date='4/10/2020', confirmed=10408, recovered=1183, ventilated=124, deceased=95),
          dict(Date='4/11/2020', confirmed=10408, recovered=1183, ventilated=124, deceased=95),
          dict(Date='4/12/2020', confirmed=11145, recovered=1627, ventilated=131, deceased=103),
          dict(Date='4/13/2020', confirmed=11586, recovered=1855, ventilated=132, deceased=116),
          dict(Date='4/14/2020', confirmed=12046, recovered=2195, ventilated=133, deceased=123),
          dict(Date='4/15/2020', confirmed=12501, recovered=2563, ventilated=133, deceased=130),
          #dict(Date='4/16/2020', confirmed=12591, recovered=2624, ventilated=140, deceased=140), # morning
          ]
    df = pd.DataFrame(li)
    df['Date']=pd.to_datetime(df['Date'])

    df_new_positives=pd.DataFrame([
              dict(Date='3/21/2020', tests=1860, conf_negatives=0, new_positives=105),
              dict(Date='3/22/2020', tests=3095, conf_negatives=0, new_positives=264),
              dict(Date='3/23/2020', tests=3743, conf_negatives=0, new_positives=345),
              dict(Date='3/24/2020', tests=5067, conf_negatives=0, new_positives=448),
              dict(Date='3/25/2020', tests=5624, conf_negatives=0, new_positives=463),
              dict(Date='3/26/2020', tests=5768, conf_negatives=0, new_positives=548),
              dict(Date='3/27/2020', tests=5513, conf_negatives=0, new_positives=393),
              dict(Date='3/28/2020', tests=5040, conf_negatives=0, new_positives=420),
              dict(Date='3/29/2020', tests=6489, conf_negatives=140, new_positives=492),
              dict(Date='3/30/2020', tests=5681, conf_negatives=94, new_positives=466),
              dict(Date='3/31/2020', tests=7851, conf_negatives=0, new_positives=741), # pdf from april 11th with only the number of tests.
              dict(Date='4/1/2020', tests=8213, conf_negatives=0, new_positives=645),
              dict(Date='4/2/2020', tests=9082, conf_negatives=0, new_positives=733),
              dict(Date='4/3/2020', tests=9903, conf_negatives=0, new_positives=632),
              dict(Date='4/4/2020', tests=6647, conf_negatives=0, new_positives=456),
              dict(Date='4/5/2020', tests=9279, conf_negatives=0, new_positives=629),
              dict(Date='4/6/2020', tests=7250, conf_negatives=0, new_positives=352),
              dict(Date='4/7/2020', tests=6592, conf_negatives=0, new_positives=343),
              dict(Date='4/8/2020', tests=5570, conf_negatives=0, new_positives=436),
              dict(Date='4/9/2020', tests=5521, conf_negatives=0, new_positives=318),
              dict(Date='4/10/2020', tests=5980, conf_negatives=0, new_positives=374),
              #dict(Date='4/11/2020', tests=7851, conf_negatives=, new_positives=),
        #dict(Date='4/11/2020', tests=7851, conf_negatives=, new_positives=),
              ])
    df_new_positives['Date'] = pd.to_datetime(df_new_positives['Date'])
    #print(df_new_positives)
    #print(df.diff(axis=0))
    df = pd.merge(df,df_new_positives,on='Date',how='outer')
    return df

def logistic(x,L,k,x0):
    coef = np.clip(-k*(x-x0),a_min=-np.inf,a_max=100)
    y = L/(1+np.exp(coef))
    return y

def fit_arbitrary_curve(X,y,fun=logistic,guess=(50,10,2)):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    params, pcov = curve_fit(fun, X, y, guess,maxfev=10000)
    pred = fun(X, *params)
    R2 = 1.-np.var(pred - y)/np.var(y)
    return R2, params



def daily_analysis(df, mpl=False, tar_day=15,tar_month=4):
    types = 'confirmed','recovered','ventilated','deceased'
    start_date = '3/5/2020'

    if mpl:
        ax_dilute = 3
        sp=2
        _, axes = plt.subplots(sp,sp)
    df0 = df.copy()
    for i, typ in enumerate(types):
        df = df0.copy()
        if typ == 'deceased':# or typ == 'ventilated':
            # take only from first death
            start_date = df.loc[df['deceased']>0]['Date'].iloc[0]
            start_date = '{:02d}/{:02d}/2020'.format(start_date.month,start_date.day)
        else:
            start_date = '3/5/2020'
        df = df.query('Date > "{}"'.format(start_date))
        epoch = datetime.datetime.utcfromtimestamp(0)
        df['date_i'] = df['Date'].apply(lambda x: (x-epoch).total_seconds())
        dates = df['date_i'].to_numpy()
        final_date_i = (datetime.datetime(day=tar_day, month=tar_month, year=2020) - epoch).total_seconds()
        more_dates = np.round(np.linspace(dates[-1], final_date_i, 10))
        dates_i = np.concatenate((dates, more_dates))
        dates_date = [datetime.datetime.fromtimestamp(x) for x in dates_i]

        X,y=df['date_i'].to_numpy()[...,None],df[typ].apply(lambda x: np.log(1+x)).to_numpy()
        log_reg = LinearRegression().fit(X,y)
        log_pred = log_reg.predict(dates_i[...,None])
        log_pred = np.exp(log_pred)-1
        log_r2 = log_reg.score(X,y)

        X,y= df['date_i'].to_numpy()[..., None], df[typ].to_numpy()
        lin_reg = LinearRegression().fit(X,y)
        lin_pred = lin_reg.predict(dates_i[..., None])
        lin_r2 = lin_reg.score(X,y)

        mindate = X.min()
        X1 = X.squeeze()-mindate
        maxdate = X1.max()
        X1 /= maxdate
        logi_r2, logi_params = fit_arbitrary_curve(X1, y)
        logi_pred = logistic((dates_i[..., None].squeeze()-mindate)/maxdate, *logi_params)

        if mpl:
            ax = axes[i//sp,i%sp]
            ax.plot(df['date_i'],df[typ],lw=3)
            ax.plot(dates_i,log_pred,'--')
            ax.plot(dates_i,lin_pred, '-.')
            ax.plot(dates_i,logi_pred, ':')
            leg = ['Israel','Exp. model (R²={:.2f})'.format(log_r2),
                   'Lin. model (R²={:.2f})'.format(lin_r2),
                   'Logis. model (R²={:.2f})'.format(logi_r2),
                   ]

            ax.set_xticks(dates_i[::ax_dilute])
            ax.set_xticklabels([x.strftime('%d/%m') for x in dates_date[::ax_dilute]], rotation=45)
            ax.set_xlabel('')

            if i==0:
                ax2 = ax.twinx()
                df['diff'] = df[typ].diff()
                color = 'tab:purple'
                ax2.plot(df['date_i'],df['diff'],color=color)
                ax2.set_ylabel('Day difference', color=color)
                ax2.yaxis.set_tick_params(colors=color)
                ax2.legend(['Difference'],loc='upper right')
                ax2.set_xticks(dates_i[::ax_dilute])
                ax2.set_xticklabels([x.strftime('%d/%m') for x in dates_date[::ax_dilute]], rotation=45)
                ax2.set_ylim([0,1000])
            ax.set_title(typ)
            ax.set_ylabel('Cases')
            ax.legend(leg)

    if mpl:
        plt.tight_layout()
        plt.show()

def predict(X,y,Xpred,kind):
    if kind == 'exp':
        y = np.log(1+y)
        log_reg = LinearRegression().fit(X, y)
        log_pred = log_reg.predict(Xpred)
        pred, r2 = np.exp(log_pred) - 1 , log_reg.score(X, y)
    elif kind == 'lin':
        lin_reg = LinearRegression().fit(X, y)
        pred, r2 = lin_reg.predict(Xpred), lin_reg.score(X, y)
    elif kind == 'logi':
        mindate = X.min()
        X1 = X.squeeze() - mindate
        maxdate = X1.max()
        X1 /= maxdate
        r2, logi_params = fit_arbitrary_curve(X1, y)
        pred = logistic((Xpred.squeeze() - mindate) / maxdate, *logi_params)
    return dict(pred=pred, r2=r2)

def cumulative_ventilated_prediction(df,tar_day = 15):

    # params:
    max_reach = 1500

    types = 'confirmed','ventilated','deceased'
    mods = 'exp','logi'
    past = 11
    tar_date = datetime.datetime(year=2020, month=4, day=tar_day)
    df0 = df.copy()
    fig, axes = plt.subplots(len(types),figsize=(12,8))
    marker='d>'
    color='tab:blue','tab:orange'
    mod_labels = ['Exponential model','Logistic model']

    preds_ij={}
    for i,typ in enumerate(types):
        ax = axes[i]
        preds_ij[typ]={}
        for j,mod in enumerate(mods):
            df = df0.copy()
            if typ != 'deceased':
                start_date = '3/5/2020'
            else:
                start_date = df.loc[df['deceased'] > 0]['Date'].iloc[0]
                start_date = '{:02d}/{:02d}/2020'.format(start_date.month, start_date.day)

            df = df.query('Date > "{}"'.format(start_date))
            epoch = datetime.datetime.utcfromtimestamp(0)
            df['date_i'] = df['Date'].apply(lambda x: (x - epoch).total_seconds())
            dates = df['date_i'].to_numpy()
            final_date_i = (tar_date - epoch).total_seconds()
            more_dates = np.round(np.linspace(dates[-1] + 1, final_date_i, 10))
            dates_i = np.concatenate((dates, more_dates))
            dates_date = [datetime.datetime.fromtimestamp(x) for x in dates_i]
            int2dt = lambda arr: [datetime.datetime.fromtimestamp(x) for x in arr]
            # X0log,y0log=df['date_i'].to_numpy()[...,None],df[typ].apply(lambda x: np.log(1+x)).to_numpy()
            X0 = df['date_i'].to_numpy()[..., None]

            y0 = df[typ].to_numpy()
            kinds = 'exp','lin','logi'

            date_passes_max_reach = []
            for p in range(past):
                mx=len(X0)-p-1
                X = X0[:mx]
                y = y0[:mx]

                # predict some models
                preds = {kind:predict(X,y,dates_i[..., None],kind=kind) for kind in kinds}
                d=dict(past=datetime.datetime.fromtimestamp(dates_i[len(X)]),past_i=dates_i[len(X)])
                for k,v in preds.items():
                    # best = np.argmin( (v['pred']-max_reach)**2 )
                    # v['date_max_vent'] = dates_i[best]
                    # d[k] = datetime.datetime.fromtimestamp(dates_i[best])#.strftime('%d/%m')
                    # d['{}_i'.format(k)] = dates_i[best]  # .strftime('%d/%m')
                    # d['{}_r2'.format(k)] = v['r2']

                    # for prediction in a certain date
                    best = np.argmin((dates_i - ((tar_date-epoch).total_seconds())) ** 2)
                    d['{}_pred'.format(k)] = v['pred'][best]
                date_passes_max_reach.append(d)
            ddf = pd.DataFrame(date_passes_max_reach)
            ddf = ddf.sort_values('past')
            print(ddf.to_string())
            # start_date1 = (datetime.datetime(year=2020,month=3,day=22)-epoch).total_seconds()
            # end_date1 = (datetime.datetime(year=2020,month=4,day=14)-epoch).total_seconds()
            # pred_dates_i = np.linspace(start_date1,end_date1,20)
            # pred_dates = int2dt(pred_dates_i)
            # chg_pred_lin = predict(ddf['past_i'].to_numpy()[...,None],ddf['{}_i'.format(mod)].to_numpy(),pred_dates_i[...,None],kind='lin')
            # ax.plot(ddf['past'],ddf['{}'.format(mod)],'.--')
            # ax.plot(pred_dates, int2dt(chg_pred_lin['pred']))

            # d = 15
            # ax.text(pred_dates[d],int2dt(chg_pred_lin['pred'])[d],'R²={:.2f}'.format(chg_pred_lin['r2']))
            # ax.set_xlabel('prediction from date')
            # ax.set_ylabel('reach {} {} patients'.format(max_reach,typ))
            # ax.yaxis.set_major_formatter(DateFormatter('%d-%m'))
            # ax.xaxis.set_major_formatter(DateFormatter('%d-%m'))
            # fig.autofmt_xdate()
            preds_ij[typ][mod]=ddf['{}_pred'.format(mod)]
            ax.plot(ddf['past'], preds_ij[typ][mod], '{}-'.format(marker[j]))
            ax.set_xlabel('prediction from date')
            ax.set_ylabel('{}'.format(typ))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.xaxis.set_major_formatter(DateFormatter('%d-%m'))
            #ax.autofmt_xdate()
        today = df.iloc[-1][typ]
        ax.plot([ddf['past'].iloc[0],ddf['past'].iloc[-1]],2*[today],':',lw=3)
        # errors
        errors = {mm : np.abs(100.*(preds_ij[typ][mm]-today)/today) for mm in mods}
        for j,mm in enumerate(mods):
            fa = 2 if i==0 else 1
            for x,y,e in zip(ddf['past'],preds_ij[typ][mm],errors[mm]):
                ax.text(x,y+(-15*fa if j==0 else -15*fa),'{:.1f}%'.format(np.round(e,1)),rotation=-30,ha='center',color=color[j])
        ax.legend(mod_labels+['Today'],loc='center left')
        ax.set_title('Predicting {}\'s {} from X days in the past (% - error relative to today)'.format(tar_date.strftime('%d/%m'),typ))
    plt.tight_layout()
    plt.show()
    exit(0)


def ventilated_confirmed_correlated(df):
    start_date = '3/5/2020'
    fig, ax = plt.subplots()

    df = df.query('Date > "{}"'.format(start_date))
    epoch = datetime.datetime.utcfromtimestamp(0)
    df['date_i'] = df['Date'].apply(lambda x: (x - epoch).total_seconds())
    dates = df['date_i'].to_numpy()
    final_date_i = (datetime.datetime(day=30, month=5, year=2020) - epoch).total_seconds()
    more_dates = np.round(np.linspace(dates[-1], final_date_i, 100))
    dates_i = np.concatenate((dates, more_dates))
    dates_date = [datetime.datetime.fromtimestamp(x) for x in dates_i]
    int2dt = lambda arr: [datetime.datetime.fromtimestamp(x) for x in arr]
    # X0log,y0log=df['date_i'].to_numpy()[...,None],df[typ].apply(lambda x: np.log(1+x)).to_numpy()
    X0, y0 = df['date_i'].to_numpy()[..., None], df['ventilated'].to_numpy()
    ax.plot(df['Date'],df['confirmed'])
    ax.set_ylabel('confirmed')
    ax2=ax.twinx()
    col = 'tab:red'
    ax2.set_ylabel('ventilated', color=col)
    ax2.tick_params(axis='y', labelcolor=col)
    ax2.plot(df['Date'],df['ventilated'],color=col)
    ax2.spines["right"].set_position(("axes", 1.0))
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    col = 'tab:orange'
    ax3.set_ylabel('recovered', color=col)
    ax3.tick_params(axis='y', labelcolor=col)
    ax3.plot(df['Date'], df['recovered'], color=col)
    ax4 = ax.twinx()
    ax4.spines["right"].set_position(("axes", 1.35))
    col = 'tab:pink'
    ax4.set_ylabel('deceased', color=col)
    ax4.tick_params(axis='y', labelcolor=col)
    ax4.plot(df['Date'], df['deceased'], color=col)

    ax.xaxis.set_major_formatter(DateFormatter('%d-%m'))
    #ax.spines['right'].set_position(('axes',1.1))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
    kinds = 'exp', 'lin', 'logi'
    max_vent = 1500
    date_passes_max_vent = []
    for p in range(past):
        mx = len(X0) - p
        X = X0[:mx]
        y = y0[:mx]

        # predict some models
        preds = {kind: predict(X, y, dates_i[..., None], kind=kind) for kind in kinds}
        d = dict(past=datetime.datetime.fromtimestamp(dates_i[len(X)]), past_i=dates_i[len(X)])
        for k, v in preds.items():
            best = np.argmin((v['pred'] - max_vent) ** 2)
            v['date_max_vent'] = dates_i[best]
            d[k] = datetime.datetime.fromtimestamp(dates_i[best])  # .strftime('%d/%m')
            d['{}_i'.format(k)] = dates_i[best]  # .strftime('%d/%m')
            d['{}_r2'.format(k)] = v['r2']
        date_passes_max_vent.append(d)
    ddf = pd.DataFrame(date_passes_max_vent)
    # print(ddf)
    ddf = ddf.sort_values('past')
    # print(ddf['past_i'].to_numpy()[...,None],ddf['exp'].to_numpy(),dates_i[...,None])
    print(df)
    start_date1 = (datetime.datetime(year=2020, month=3, day=22) - epoch).total_seconds()
    end_date1 = (datetime.datetime(year=2020, month=4, day=14) - epoch).total_seconds()
    pred_dates_i = np.linspace(start_date1, end_date1, 20)
    pred_dates = int2dt(pred_dates_i)
    chg_pred_lin = predict(ddf['past_i'].to_numpy()[..., None], ddf['exp_i'].to_numpy(), pred_dates_i[..., None],
                           kind='lin')
    # chg_pred_exp = predict(ddf['past_i'].to_numpy()[..., None], ddf['exp_i'].to_numpy(), pred_dates_i[..., None], kind='exp')
    # ddf.plot(ax=ax,x='past',y=['exp'])
    ax.plot(ddf['past'], ddf['exp'], '.--')
    ax.plot(pred_dates, int2dt(chg_pred_lin['pred']))
    # ax.plot(pred_dates, int2dt(chg_pred_exp['pred']))
    d = 15
    ax.text(pred_dates[d], int2dt(chg_pred_lin['pred'])[d], 'R²={:.2f}'.format(chg_pred_lin['r2']))
    # ax.text(pred_dates[d], int2dt(chg_pred_exp['pred'])[d], 'R²={:.2f}'.format(chg_pred_exp['r2']))
    # ax.plot(dates_date, int2dt(chg_pred_exp['pred']))
    ax.set_xlabel('prediction from date')
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    ax.set_ylabel('reach {} ventilated patients'.format(max_vent))
    ax.yaxis.set_major_formatter(DateFormatter('%d-%m'))
    ax.xaxis.set_major_formatter(DateFormatter('%d-%m'))

    fig.autofmt_xdate()
    # fig.autofmt_ydate()
    plt.show()
    exit(0)


def correlate_tests_and_positives(df):
    # dict(Date='3/23/2020', tests=3743, conf_negatives=0, new_positives=345),
    df = df.dropna()
    df['Date_text'] = df['Date'].apply(lambda x:datetime.datetime.strftime(x,'%d/%m'))
    df1 = df.query('Date > "3/23/2020"')

    fig_a = px.scatter(df, x="tests", y="new_positives", text='Date_text', log_x=False, trendline='ols')  # , size_max=60)
    fig_b = px.scatter(df1, x="tests", y="new_positives", text='Date_text', log_x=False, trendline='ols')
    fig_a.data[0]['marker']['size'] = 14
    fig_a.data[0]['marker']['color'] = 'blue'
    fig_b.data[1]['marker']['color'] = 'red'
    # ugly
    sel_r2 = float(re.findall('>R<sup>2<\/sup>=([\d\.]+)<br>', fig_b.data[1]['hovertemplate'])[0])
    all_r2 = float(re.findall('>R<sup>2<\/sup>=([\d\.]+)<br>', fig_a.data[1]['hovertemplate'])[0])
    fig = go.Figure()
    fig.add_trace( fig_a.data[1] )
    fig.add_trace(fig_a.data[0])
    fig.add_trace(fig_b.data[1])
    d = 170
    da = 2
    db = 2
    fig.update_traces(textposition='top center')
    fig.add_trace(go.Scatter(
        x=[fig_a.data[1]['x'][da]+d, fig_b.data[1]['x'][db]+d],textfont=dict(color=['blue','red']),
        y=[fig_a.data[1]['y'][da], fig_b.data[1]['y'][db]],
        mode="text",        legendgroup='',  textposition="bottom center",
        text=['R²={:.2f}'.format(all_r2), 'R²={:.2f}'.format(sel_r2)],
    ))

    fig.update_layout(
        height=700, width=900,
        xaxis_title='Tests',
        yaxis_title='New positives',
        title_text='Number of daily tests against number of new confirmed positives',
    )

    fig.show()
    fig = go.Figure(data=[
        go.Bar(name='Tests', x=df['Date'], y=df['tests']),
        go.Bar(name='New positives', x=df['Date'], y=df['new_positives']),
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.show()

def daily_confirmed_city_analysis(df):
    # dict(Date='3/23/2020', tests=3743, conf_negatives=0, new_positives=345),
    df_c = df_city_confirmed.copy()

    #df = df.dropna()
    print(df_c.to_string())
    df['Date_text'] = df['Date'].apply(lambda x:datetime.datetime.strftime(x,'%d/%m'))
    #df_c['Date_text'] = df_c['Date'].apply(lambda x:datetime.datetime.strftime(x,'%d/%m'))
    cities = [x for x in df_c.columns if x != 'Date']
    # ord_cities = ['bnei_brak','jerusalem']
    # cities = [x for x in cities if x not in ord_cities]+ord_cities
    for c in cities:
        df[c]=0
    for i,(k,row) in enumerate(df.iterrows()):
        #print(row['Date'], df_c['Date'].tolist())
        if row['Date'] in df_c['Date'].tolist():
            c_cur = df_c.query('Date == "{}"'.format(row['Date'])).drop(['Date'],axis=1)
            all1 = c_cur[cities].sum(axis=1)
            row['confirmed'] -= all1
            for c in cities:
                row[c] = c_cur[c]
            df.iloc[i] = row

    print(df.to_string())
    data = [go.Bar(name='Confirmed', x=df['Date'], y=df['confirmed'])]

    data.extend([go.Bar(name='{}'.format(c), x=df['Date'], y=df[c]) for c in cities])
    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(barmode='stack',yaxis_title='Confirmed',legend=dict(font=dict(size=20)))

    fig.show()

def daily_confirmed_city_single(df,city='kiryat_yam'):
    # dict(Date='3/23/2020', tests=3743, conf_negatives=0, new_positives=345),
    df_c = df_city_confirmed.copy()

    df['Date_text'] = df['Date'].apply(lambda x:datetime.datetime.strftime(x,'%d/%m'))

    #cities = [x for x in df_c.columns if x != 'Date']
    # ord_cities = ['bnei_brak','jerusalem']
    print(df_c.to_string())
    cities = [city]
    # cities = [x for x in cities if x not in ord_cities]+ord_cities
    for c in cities:
        df[c] = 0
    for i,(k,row) in enumerate(df.iterrows()):
        #print(row['Date'], df_c['Date'].tolist())
        if row['Date'] in df_c['Date'].tolist():
            c_cur = df_c.query('Date == "{}"'.format(row['Date'])).drop(['Date'],axis=1)
            all1 = c_cur[cities].sum(axis=1)
            row['confirmed'] -= all1
            for c in cities:
                row[c] = c_cur[c]
            df.iloc[i] = row
    df = df.query('Date > "3/29/2020"')
    df = df.drop('confirmed',axis=1)
    #print(df.to_string())
    data = [go.Bar(name=city, x=df['Date'], y=df[city])]

    #data.extend([go.Bar(name='{}'.format(c), x=df['Date'], y=df[c]) for c in cities])
    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(barmode='stack',title=city,yaxis_title='Confirmed',yaxis=dict(titlefont=dict(size=20)))

    fig.show()

def estimate_confirmed_for_test_num(df):
    # just some experiment, did not pan out
    df = df.dropna()
    df['Date_text'] = df['Date'].apply(lambda x: datetime.datetime.strftime(x, '%d/%m'))
    # df1 = df.query('Date > "3/23/2020"')
    pp = {0: df.query('tests < 3500'),
          1: df.query('tests >=3500'),}# and tests < 7500'),
      #2: df.query('tests >= 7500')}
    a, b = 'tests', 'new_positives'
    grp1_rat = np.mean(1. * pp[1][b] / pp[1][a])
    grp1_std = np.std(1.*pp[1][b]/pp[1][a])
    ratios = np.array([grp1_rat, grp1_rat-grp1_std, grp1_rat+grp1_std])
    print('ratios: {}'.format(ratios))
    est_line_x = np.array([df['tests'].min(), df['tests'].max()])
    est_line_y = np.outer(ratios,est_line_x)
    markers = '-','--',':'
    #gs1 = gridspec.GridSpec(3, 1)
    #gs1.update(wspace=0.025, hspace=0.05)
    fig,ax=plt.subplots(3,figsize=(7,8))

    for i in range(len(pp)):
        ax[0].scatter(pp[i][a],pp[i][b])
    # correlate estimates
    for i, ys in enumerate(est_line_y):
        ax[0].plot(est_line_x,ys,markers[i],color='tab:green')
    leg = ['Mean est.','Lower est.','Higher est.']
    ax[0].legend(leg)
    ax[0].set_xlabel('No. of tests')
    ax[0].set_ylabel('Daily positives')
    ax[0].set_title('New positives v. number of daily tests')
    start_date = datetime.datetime(2020,3,23).strftime('%m/%d/%Y')
    init_confirmed = int(df.query('Date == "{}"'.format(start_date))['confirmed'])
    df_rest = df.query('Date > "{}"'.format(start_date))
    agg_positives = init_confirmed + df_rest['new_positives'].cumsum()
    ax[1].plot(df['Date_text'],df['confirmed'],'.-',lw=2)
    #ax[1].plot(df_rest['Date_text'],agg_positives)
    for i, rat in enumerate(ratios):
        est = init_confirmed+(df_rest['tests']*rat).cumsum()
        ax[1].plot(df_rest['Date_text'],est,markers[i])
    #ax[1].legend(['Raw','aggregated daily positives']+leg)
    ax[1].legend(['Raw'] + leg)
    ax[1].set_xticklabels(df['Date_text'], rotation=45)
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Confirmed')
    ax[1].set_title('Raw confirmed cases v. estimates')
    #print(df_rest['tests'].mean())
    # let's see what we would observe if the number of tests were different.
    n_tests = np.array([3000, 5000, 7000, 9000])
    ww = 200
    rr = [0,-1,1]
    colors = 'tab:orange','tab:green','tab:red'
    true = init_confirmed + df_rest['new_positives'].cumsum()
    for i, rat in enumerate(ratios):
        dat = []
        for j, n in enumerate(n_tests):
            est = init_confirmed + ((df_rest['tests']*0+n) * rat).cumsum()
            dat.append(est.iloc[-1])
        dat = np.array(dat)
        ax[2].bar(n_tests+ww*rr[i], dat,width=ww,color=colors[i])
    #ax[2].bar(n_tests+ww*2, dat*0+true.iloc[-1],width=ww,color='tab:blue')
    ax[2].plot([n_tests[0]-ww*2,n_tests[-1]+ww*2],2*[true.iloc[-1]],'--',color='tab:blue')
    ax[2].set_xticks(n_tests)
    ax[2].legend(['Confirmed']+leg)
    ax[2].set_xlabel('Number of tests')
    ax[2].set_ylabel('Estimated confirmed cases')
    ax[2].set_yticks(range(1000,20000,4000))
    ax[2].grid('minor')
    ax[2].set_title('Estimated confirmed cases by number of tests')
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    df = manual_corona_israel()
    # daily_analysis(df,mpl=True,tar_day=20)
    # daily_confirmed_city_analysis(df)
    # daily_confirmed_city_single(df,city='kiryat_yam')
    # get_haifa_confirmed_rank()
    cumulative_ventilated_prediction(df,tar_day=15)
    #estimate_confirmed_for_test_num(df)
    # tests_per_city()

    # correlate_tests_and_positives(df)
    # deceased_df = get_deceased_stats()

    # ventilated_confirmed_correlated(df)