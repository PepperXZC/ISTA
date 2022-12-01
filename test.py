import requests
headers={
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400'}
url='https://manhuabika.com/pchapter/?cid=5f7c91eea3baf879dfb9e266&chapter=1&chapterPage=1&maxchapter=1'
re=requests.get(url,headers=headers)
def get_pictures_urls(text):
    st='img src="'
    m=len(st)
    i=0
    n=len(text)
    urls=[]#储存url
    while i<n:
        if text[i:i+m]==st:
            url=''
            for j in range(i+m,n):
                if text[j]=='"':
                    i=j
                    urls.append(url)
                    break
                url+=text[j]
        i+=1
    return urls
urls=get_pictures_urls(re.text)
for url in urls:
    print(url)
