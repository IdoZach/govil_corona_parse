import pandas as pd
#from html.parser import HTMLParser
#from collections import defaultdict
#import urllib.request
import tempfile
import pickle
import requests
from lxml import etree
import shutil
import os.path as osp
from pprint import pprint
import datetime
import re
import os

class Parser(object):
    def __init__(self,queries=('חיפה',)):

        #self.debug = True
        self.debug = False
        if type(queries) is str:
            queries = (queries,)
        self.queries = queries
        self.addr_prefix = 'https://www.gov.il/he/departments/news/'

        save_dir = './' #
        s_queries = '_'.join(queries)
        self.tar_fname = osp.join(save_dir,'govil_parse_{}_{}'.format(datetime.datetime.now().strftime('%d%m%y_%Hh'),s_queries))
        self.tar_fname_recent = osp.join(save_dir,'govil_parse_{}_{}'.format('recent', s_queries))
        print('target file name: {}'.format(self.tar_fname))
        self.res_data = None
    def get_addresses(self):
        start_date = datetime.datetime(day=1,month=3,year=2020)
        end_date = datetime.datetime.now()
        delta = datetime.timedelta(days=1)
        cur = start_date
        err_msg = 'לא מצאנו את מה שחיפשת.'
        #addresses = []
        html = None
        while cur <= end_date:
            dd = cur.day
            mm = cur.month
            yyyy = cur.year
            for num in range(1,100): # no more than 100 updates per day
                url = '{}{:02d}{:02d}{:04d}_{:02d}'.format(self.addr_prefix, dd,mm,yyyy, num)
                success = False
                seek_next_page = False
                for tries in range(3):
                    try:
                        print('try {}, checking {}...'.format(tries, url))
                        html = requests.get(url).text
                        if err_msg in html:
                            print('no page')
                            seek_next_page = True
                            pass
                        else:
                            success = True
                        break
                    except Exception as e:
                        print(e)
                if success:
                    yield html, url
                    #addresses.append(url)
                if seek_next_page:
                    break

            cur += delta
        #return addresses

    def parse_details(self,s=None,url='',patient=-1):
        date = re.findall('\d\d\.\d\d\.\d\d\d\d',s)
        hours = re.findall('\d\d:\d\d',s)
        return dict(date=date,hours=hours,text=s,url=url,patient=patient)

    def read_page(self, html=None, addr='https://www.gov.il/he/departments/news/20032020_04'):
        if html is None:
            html = requests.get(addr).text
            debug=True
        else:
            debug=False
        dom = etree.HTML(html)
        text = dom.xpath('//div[@id="NewsContent"]/p/node()')
        dat = []
        cur_patient=-1
        for line in text:
            if type(line) is not etree._Element: # i.e. string with details
                if any([q in line for q in self.queries]):
                    di = self.parse_details(line,addr,patient=cur_patient)
                    print('patient: {}, date: {}, times: {}, all: {}'.format(di['patient'], di['date'], di['hours'], di['text']))

                    dat.append(di)
            else: # element
                try:
                    cur_patient = re.findall('{} (\d+)'.format('חולה מספר'),line.text)[0]
                except:
                    pass
                #    cur_patient = -1
        return dat
    def read_pages(self):
        data = []
        for html, url in self.get_addresses():
            data.extend(self.read_page(html=html, addr=url))
        self.res_data = data
    def save_parsed_data(self):
        data = self.res_data
        df = pd.DataFrame(data)
        df['text'] = df['text'].apply(str.strip)
        df['url'] = df['url'].apply(lambda x: '<a href={}>{}</a>'.format(x,x))
        df = df.iloc[::-1] # reverse
        open(self.tar_fname+'.html','w').writelines(df.to_html(escape=False))
        open(self.tar_fname+'.md', 'w').writelines(df.to_markdown())
        print('written HTML to {}'.format(self.tar_fname+'.html'))
        print('written MD to {}'.format(self.tar_fname+'.md'))
        shutil.copy(self.tar_fname+'.md',self.tar_fname_recent+'.md')
        os.system('xdg-open {}'.format(self.tar_fname+'.html'))

def someplot():
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.cumsum(np.random.randn(1000))
    print(x)
    plt.plot(x)
    plt.show()
    exit(0)
if __name__ == '__main__':
    #someplot()
    #parser = Parser(queries='חיפה')
    parser = Parser(queries=['חיפה','קרית ים','קריית ים','אתא','מוצקין','קרית חיים','קריית חיים','ביאליק','הקריות'])
    # parser = Parser(queries='סטוק')
    # parser = Parser(queries='כנסת')
    # parser = Parser(queries=['יקנעם','יוקנעם'])
    # parser.read_page()
    parser.read_pages()
    parser.save_parsed_data()
