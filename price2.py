import re
import linkGrabber
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

def findprice(found2): 
	
	url="http://www.mysmartprice.com/"+found2
	print (url)
	#time.sleep(200)
	r=requests.get(url)
	p=r.content
	f1=open("page.html","w")
	f1.write(str(r.content))
	f2=open("page.html","r")
	f1.close()
	soup = BeautifulSoup(f2.read())
	f2.close()
	string= soup.encode("utf-8")
	#print(soup.encode("utf-8"))
	#time.sleep(400)
	pattern='<div class="store_pricetable online_line"(.*?)<div'
	key=re.findall(pattern,str(string))
	print(key)
	pattern='data-pricerank="(.*?)"'
	price=re.findall(pattern,str(string))
	pattern1='data-storename="(.*?)">'
	site=re.findall(pattern1,str(key))
	print(price)
	print(site)
	table = [site, price]
	df = pd.DataFrame(table)
	df = df.transpose()
	cols = ['Site', 'Price']
	df.columns = cols
	return df
	 

def getprice(product):
	site ="mysmartprice"  
	
	start="'href': '/url?q="  
	end="&sa" 
	links = linkGrabber.Links('https://www.google.co.in/search?newwindow=1&biw=1366&bih=659&q=' + site +'+'+product)
	gb = links.find(limit=30)
	print(gb[25])
	gb1=str(gb)
	frame = pd.DataFrame()
	gb2 = re.search("http://www.mysmartprice.com/(.+?)(%|&)",gb1)
	#print(gb2.group(1))
	if gb2:
		found2=gb2.group(1)
		print(found2)
		frame=findprice(found2)
        
	else:
		print('Not found')
	return frame
#getprice()