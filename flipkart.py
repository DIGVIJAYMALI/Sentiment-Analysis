import re
import linkGrabber
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import nltk
import threading, random
#nltk.download()
import textblob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import wordnet
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from nltk.corpus import movie_reviews
import matplotlib.pyplot as plt
import statistics 
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

def scrap(allLinks,x):
	Review_Frame=pd.DataFrame()
	date_frame=pd.DataFrame({})
	frame = pd.DataFrame()
	all=[]
	all_dates=[]

	for num in range(1,1,1):
		singlelink=allLinks+x[0]+"?page="+str(num)+'&&pid='+x[1]
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``")
		print(singlelink)
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``")
		#time.sleep(500)
		r=requests.get(singlelink)
		p=r.content
		#print(r.content)
		f1=open("page.html","w")
		f1.write(str(r.content))
		f2=open("page.html","r")
		f1.close()
		soup = BeautifulSoup(f2.read())
		print("~~~~~~~~~~~~~~~~`page.html~~~~~~~~~~~~~~~")
		print(soup)

		f2.close()
		#titles=soup.find_all("div", {"class": "line fk-font-normal bmargin5 dark-gray"})
		#fp_titles=open("titles-1.txt","a+")
		#fp_titles.write(str(titles))
		review_data= soup.find_all("div", {"class": "qwjRop"})
		date_data = soup.find_all("p", {"class": "_3LYOAd"})
		pattern_date='<div class="date line fk-font-small">(.*?)</div>'
		date_a=re.findall(pattern_date,str(date_data))
		all_dates=all_dates+date_a[1:]
		#f3=open("reviews-1.txt","a+")
		#f3.write(str(review_data))
		print("~~~~~~~~~~~~~`~~~~~~~~~~~~~~~`review data~~~~~~~~~~~~~~~~~~~~~~~`````````````")
		print(review_data)
		
		pattern=', <span class="review-text">(.*?)</span>'
		review=re.findall(pattern,str(review_data))
		review=[r.replace('\\n','').replace('<br/>','') for r in review]
		print("~~~~~~~~~~~~~`~~~~~~~~~~~~~~~`review frmae~~~~~~~~~~~~~~~~~~~~~~~`````````````")
		print(review)
		print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
		review_dir={'Review':review}
		review_dir_frame=pd.DataFrame(review_dir)
		print(" DATAFRAME SMALL-->  ")
		print(review_dir_frame)
		Review_Frame=Review_Frame.append(review_dir_frame,ignore_index=True)
	for s in all_dates:
		date_object = datetime.strptime(s , '\\n %d %b %Y ')
		date_frame=date_frame.append({'Date':date_object},ignore_index=True)
	print('*****date****')
	print (date_frame)
	#time.sleep(500)
	print(" FINAL DATAFRAME -->  ")
	print(Review_Frame)
	print(len(Review_Frame.index))
	Target_Word_Frame=pd.DataFrame()
	tknzr = TweetTokenizer()
	stop_words=set(stopwords.words("english"))
	#time.sleep(5)
    #ADD LOOP HERE FOR ALL REVIEWS FOR THAT PRODUCT
	print("WAIT FOR DEFINING ALGO AND TRAINING")
	def word_feats(words):
		return dict([(word, True) for word in words])
 
	negids = movie_reviews.fileids('neg')
	posids = movie_reviews.fileids('pos')
 
	negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
	posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    #MODIFY THIS~~~~~~~~~~~~~~~~~~~~~~
    #negcutoff = int(len(negfeats)*0.1)
    #poscutoff = int(len(posfeats)*0.1)
 
	trainfeats = posfeats[:20]+negfeats[:20]  
	cl = NaiveBayesClassifier(trainfeats)
	REVIEW_WITHOUT_STOPWORDS=[]
	print("TRAINING COMPLETE")
	print("ANALYZING.....")
	val1=[]
	for index,row in Review_Frame.iterrows():
		REVIEW_TOKENIZE=tknzr.tokenize(row['Review'])
		count=-1
		for w in REVIEW_TOKENIZE:
			if w not in stop_words and wordnet.synsets(w) and w is not 'n':
				prob_dist = cl.prob_classify(w)
				positive=round(prob_dist.prob("pos"), 2)
				val=positive*100
				count += 1
        #DEPENDS ON TRAINING DATA 
				if val in range(0,35,1):
					#synon = wordnet.synsets(w)
					#if synon:
					val1.insert(count,val*-1)
					Target_Word_Frame=Target_Word_Frame.append({'Target_Word':w.lower(),'Polarity':val,'Tag':'NEGATIVE'},ignore_index=True)
				elif val in range(36,45,1):
					#synon = wordnet.synsets(w)
					#if synon:
					val1.insert(count,val)
					Target_Word_Frame=Target_Word_Frame.append({'Target_Word':w.lower(),'Polarity':val,'Tag':'CONFLICT'},ignore_index=True)
				else:
					#synon = wordnet.synsets(w)
					#if synon:
					val1.insert(count,val)
					Target_Word_Frame=Target_Word_Frame.append({'Target_Word':w.lower(),'Polarity':val,'Tag':'POSITIVE'},ignore_index=True)			
		
		average=statistics.mean(val1)
		if average<50.0:
			frame=frame.append({'Review':row['Review'],'Tag':'NON-SATISFACTORY','Rate':average},ignore_index=True)
		else:
			frame=frame.append({'Review':row['Review'],'Tag':'SATISFACTORY','Rate':average},ignore_index=True)
		
	print("___________________________REVIEW_TOKENIZE__________________________")
	print(Target_Word_Frame)
		#for index,row in Review_Frame.iterrows():
		#polarity=sentiment_analysis.Splitter(row['Review'])
		#print("^^^^^^^^^^^^^^AVG POL^^^^^^^^^^^^^^^^^^^^^\n")
		#print(polarity)
		#print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
			
		
		#f3.write(str(review))
		#f3.write("\nneww\n")
		#f3.close()
		#source=re.findall(r'<div class="a-row review-data"><span class="a-size-base review-text">(.*?)</span></div>',review_data)
		#print("^^^^^^^^^^^^^^^^SOURCE^^^^^^^^^^^^^^^^^^^\n")
		#print(source)
	TOTAL=len(Target_Word_Frame.index)	
	RESULT_POS=(len(Target_Word_Frame[(Target_Word_Frame.Tag == 'POSITIVE')])/TOTAL)*100
	RESULT_NEG=(len(Target_Word_Frame[(Target_Word_Frame.Tag == 'NEGATIVE')])/TOTAL)*100
	RESULT_CON=(len(Target_Word_Frame[(Target_Word_Frame.Tag == 'CONFLICT')])/TOTAL)*100
	
	RESULT_POS_N = round(RESULT_POS, 2)
	RESULT_NEG_N = round(RESULT_NEG, 2)
	RESULT_CON_N = round(RESULT_CON, 2)

	image = Image.new("RGBA", (300,350), (255,255,255))
	draw = ImageDraw.Draw(image)
	fontsize = 30
	font = ImageFont.truetype("arial.ttf", fontsize)

	draw.text((10, 0),'\n\n' + "POSITIVE : "+ str(RESULT_POS_N) + '\n'+ '\n' + "NEGATIVE : "+ str(RESULT_NEG_N) + '\n' + '\n' + "CONFLICT : "+ str(RESULT_CON_N) , (0,0,0), font = font)
	img_resized = image.resize((188,10), Image.ANTIALIAS)

	image.save("C:\\Users\\DIGVIJAY MALI\\Desktop\\PYTHON FOR ALL\\PROJECTS\\mysite\\webapp\\static\\images\\fpimg.png")

	print("~~~~~~~~~~~~~~~~RECOMMENDATION~~~~~~~~~~~~~~~~~~~\n")
	print("-------------------EXCELLENT----------------------\n")
	print(RESULT_POS)
	print("-------------------POOR----------------------\n")
	print(RESULT_NEG)
	print("-------------------SATISFACTORY----------------------\n")	
	print(RESULT_CON)
	print("-------------------TESTING DATA ANALYSIS---------------------------------\n")
	print(Target_Word_Frame['Polarity'].describe())
	print("-------------------POLARITY MEDIAN---------------------------------\n")
	print(Target_Word_Frame.Polarity.median())
	print("-------------------POLARITY standard deviation---------------------------------\n")
	print(Target_Word_Frame.Polarity.std())
	print("-------------------REVIEWS HISTOGRAM---------------------------------\n")
	frame = pd.concat([frame, date_frame], axis=1)
	frame=frame.sort_values('Date', ascending=1)
	x_y=frame[['Date','Rate']]
	
	
	fig=x_y.plot(x='Date', y='Rate',title="Flipkart Monthly Analysis")
	plt.xlabel("Months")
	plt.ylabel("Rate")
	fig1 = plt.gcf()
	plt.show()
	
	fig1.savefig('webapp/static/images/fpMonthly.png')
	
	Target_Word_Frame.Tag.value_counts().plot(kind='barh')
	plt.title('REVIEW TYPES')
	plt.xlabel('Frequency')
	fig=plt.gcf()
	plt.show()
	fig.savefig('webapp/static/images/fpbar.png')
	Target_Word_Frame.Tag.value_counts().plot(kind='pie')
	plt.axis('equal')
	plt.title('Number of appearances in dataset')
	fig=plt.gcf()
	plt.show()
	fig.savefig('webapp/static/images/fppie.png')
	#ax = s.hist()  # s is an instance of Series
	#fig = ax.get_figure()
	#fig.savefig('/path/to/figure.pdf')
	print("-------------------REVIEWS BOXPLOT---------------------------------\n")
	#plt.show(Target_Word_Frame.boxplot(column='Polarity'))
	print("@@@@@@@@@@@@@@@@@@@@@@@ END @@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
	
	return frame
	
def getlink(product):
	site ="flipkart"
	start="'href': '/url?q="
	end="&sa"
	links = linkGrabber.Links('https://www.google.co.in/search?newwindow=1&biw=1366&bih=659&q='+site+'+'+product+'+product+reviews')
	gb = links.find(limit=100)
	gb1=str(gb)
	print(gb1)
	frame = pd.DataFrame()
	f=open("link2.txt","w+")
	f.write(str(gb1))
	gb3=re.search("www.flipkart.com(.+?)/p",gb1)
	print("*******")
	link1=gb3.group(1)
	print(link1)
	print("*******")
	gb2 = re.search("product-reviews/(.+?)&",gb1)
	print(gb2.group(1))
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	#time.sleep(400)
	if gb2:
		link2= gb2.group(1)
		#link="https://www."+site+".com"+found
		print("!!!!!!!!!!!@@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@!!!!!!!!!!!!!!")
		print(link2)
		x=[]
		x=link2.split("%3Fpid%3D")
		print(x[1])
		#link='https://www.flipkart.com'+link1+'/product-reviews/'+link2
		link='https://www.flipkart.com'+link1+'/product-reviews/'

		print("~~~~~~~~~~~~~~~~~print who;e link~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		print(link)
		frame=scrap(link,x)
	else :
		print ('not found')
	return frame

getlink("sony c5")
