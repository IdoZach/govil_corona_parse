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


def corona_israel_city():
    df = pd.DataFrame([
        # this is always in the morning
        dict(Date='3/31/2020', jerusalem=650, bnei_brak=571, ashkelon=114, haifa=67, tel_aviv=278),
        dict(Date='4/1/2020', jerusalem=807, bnei_brak=723, ashkelon=125, haifa=81, tel_aviv=301),
        dict(Date='4/3/2020', jerusalem=1132, bnei_brak=1061, ashkelon=170, haifa=96, tel_aviv=337),
        dict(Date='4/5/2020', jerusalem=1302, bnei_brak=1214, ashkelon=191, haifa=105, tel_aviv=359),
        dict(Date='4/6/2020', jerusalem=1424, bnei_brak=1323, ashkelon=207, haifa=106, tel_aviv=387),
        dict(Date='4/7/2020', jerusalem=1464, bnei_brak=1386, ashkelon=209, haifa=108, tel_aviv=393),

    ])
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df_city_confirmed = corona_israel_city()

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

          # dict(Date='4/07/2020', confirmed=9004, recovered=683, ventilated=113, deceased=59), # todo  8am
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

def cumulative_ventilated_prediction(df):

    # params:
    max_reach = 1500

    types = 'ventilated','deceased'
    mods = 'exp','logi'
    past = 10
    tar_date = datetime.datetime(year=2020, month=4, day=8)
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
        ax.set_title('Predicting today\'s {} from X days in the past (% - error relative to today)'.format(typ))
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

if __name__ == '__main__':
    df = manual_corona_israel()
    daily_analysis(df,mpl=True,tar_day=15)
    daily_confirmed_city_analysis(df)

    # correlate_tests_and_positives(df)
    cumulative_ventilated_prediction(df)
    # ventilated_confirmed_correlated(df)