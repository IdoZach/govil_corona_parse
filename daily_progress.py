import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, LogisticRegression

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
          ]
    df = pd.DataFrame(li)
    df['Date']=pd.to_datetime(df['Date'])
    return df

def logistic(x,L,k,x0):
    coef = np.clip(-k*(x-x0),a_min=-np.inf,a_max=100)
    y = L/(1+np.exp(coef))
    return y

def fit_arbitrary_curve(X,y,fun=logistic,guess=(50,10,2)):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    params, pcov = curve_fit(fun, X, y, guess)
    pred = fun(X, *params)
    R2 = 1.-np.var(pred - y)/np.var(y)
    return R2, params



def daily_analysis(df, mpl=False):
    types = 'confirmed','recovered','ventilated','deceased'
    start_date = '3/5/2020'

    if mpl:
        ax_dilute = 3
        sp=2
        _, axes = plt.subplots(sp,sp)

    for i, typ in enumerate(types):
        if typ == 'deceased':
            # take only from first death
            start_date = df.loc[df[typ]>0]['Date'].iloc[0]
            start_date = '{:02d}/{:02d}/2020'.format(start_date.month,start_date.day)
        df = df.query('Date > "{}"'.format(start_date))
        epoch = datetime.datetime.utcfromtimestamp(0)
        df['date_i'] = df['Date'].apply(lambda x: (x-epoch).total_seconds())
        dates = df['date_i'].to_numpy()
        final_date_i = (datetime.datetime(day=1, month=4, year=2020) - epoch).total_seconds()
        more_dates = np.round(np.linspace(dates[-1], final_date_i, 5))
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

if __name__ == '__main__':
    df = manual_corona_israel()
    daily_analysis(df,mpl=True)