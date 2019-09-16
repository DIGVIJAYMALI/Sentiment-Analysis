import re
import linkGrabber
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

def scrap(found2):
	specification = pd.DataFrame()
	url="http://www.flipkart.com"
	print("in sxrap")
	print (url+found2)
	#time.sleep(500)
	singlelink=url+found2
	r=requests.get(singlelink)
	p=r.content
	f1=open("page.html","w")
	f1.write(str(r.content))
	f2=open("page.html","r")
	f1.close()
	soup = BeautifulSoup(f2.read())
	f2.close()
	key= soup.find_all("td", {"class":"specsKey"})
	pattern='<td class="specsKey">(.*?)</td>'
	keys=re.findall(pattern,str(key))
	value= soup.find_all("td", {"class":"specsValue"})
	pattern='<td class="specsValue">(.*?)</td>'
	values1=re.findall(pattern,str(value))
	values=[r.replace('\\n','').replace('\\t','').replace(' ','').replace('<p>','').replace('</p>','') for r in values1]
	print("*************keys**********")
	print(keys)
	print("*************values**********")
	print(values)
	table = [keys, values]
	df = pd.DataFrame(table)
	df = df.transpose()
	cols = ['Key', 'Value']
	df.columns = cols
	
	return df

def getlink(txt):
	site ="flipkat"  
	product =txt
	start="'href': '/url?q="  
	end="&sa" 
	links = linkGrabber.Links('https://www.google.co.in/search?newwindow=1&biw=1366&bih=659&q='+site+'+'+product+'+product+details')
	gb = links.find(limit=100)
	print(gb[25])
	gb1=str(gb)
	frame = pd.DataFrame()
	gb2 = re.search("http://www.flipkart.com(.+?)(%|&)",gb1)

	if gb2:
		found2=gb2.group(1)
		print(found2)
		#time.sleep(500)
		frame=scrap(found2)
        
	else:
		print('Not found')
	return frame
    

#getlink(txt)
