import requests
from bs4 import BeautifulSoup
import codecs
import  time

def get_html(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = 'utf-8'
        return r.text
    except:
        return('something is wrong.')

def get_content(url):
    comments = []#建立comments列表存储comment字典
    html = get_html(url)
    soup = BeautifulSoup(html,'lxml')#lxml解析器创建soup对象
    litags =soup.find_all('li',class_ = ' j_thread_list clearfix')
    for li in litags:
        try:#用异常
            comment = {}#循环每次更新comment，才会不被覆盖

            comment['title'] = li.find('a',class_= 'j_th_tit')['title']#标签怎么引用好好思考一下
            comment['link'] ='http://tieba.baidu.com'+li.find('a',class_ = 'j_th_tit')['href']
            comment['replynum'] = li.find('span',class_= 'threadlist_rep_num center_text').string
            comment['author'] = li.find('span',class_= 'tb_icon_author')['title']
            comment['last answer'] = li.find('span',class_ = 'tb_icon_author_rely j_replyer')['title']
            comment['last answer time'] = li.find('span', class_='threadlist_reply_date pull_right j_reply_data').string
            comments.append(comment)
        except:
            print ('there is something wrong.')
    return comments

def Out2file(dict):
    f = codecs.open('D:\\蔡徐坤贴吧爬虫.txt','w','UTF-8')#出现乱码就是codecs,'utf-8的问题
    for comment in dict:
        f.write("标题 = "+ comment['title'] + ' \t' +'链接 = ' + comment['link'] + ' \t' +
                '回复次数 = '+comment['replynum'] +' \t' + '作者 = ' + comment['author'] + ' \t' +
                '最后回复 = ' + comment['last answer']+ ' \t'+'最后回复时间='+comment['last answer time']+'\r\n')
        print('当前页面爬取完成')

def main(baseurl,deep):
    url_list = []
    for i in range(0,deep):
        url_list.append(baseurl + '&pn=' + str(50*i))
    print('所有页面已经下载到本地')
    for url in url_list:
        content = get_content(url)
        Out2file(content)
    print('所有信息已经保存完毕')

baseurl ='http://tieba.baidu.com/f?kw=%E8%94%A1%E5%BE%90%E5%9D%A4&ie=utf-8'
deep = 4
main(baseurl,deep)




