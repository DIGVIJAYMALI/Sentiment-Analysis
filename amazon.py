import re
import linkGrabber
import requests
from bs4 import BeautifulSoup
import time
import pickle
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
def scrap(found2):
	#pd.set_option('display.width',1000)
	pd.options.display.max_colwidth = 200
	url="http://www.amazon.com/product-reviews/"
	part1="/ref=cm_cr_pr_btm_link_"
	part2="?ie=UTF8&showViewpoints=1&sortBy=recent&pageNumber="
	Review_Frame=pd.DataFrame()
	all_dates=[]
	frame = pd.DataFrame()
	all=[]
	date_frame=pd.DataFrame({})
	review_rate=[]
	for num in range(1,3,1):
		
		all_pol=[]
		singlelink=url+found2+part1+str(num)+part2+str(num)
		print("^^^^^^^^^^^^^^^^link^^^^^^^^^^^^^^^^^^^\n")
		print(singlelink)
		'''#time.sleep(500)
		print("^^^^^^^^^^^^^^^^link end^^^^^^^^^^^^^^^^^^^\n")'''
		r=requests.get(singlelink)		
		p=r.content
		print(r.content)
		print("^^^^^^^^^^^^^^^^link requested content end^^^^^^^^^^^^^^^^^^^\n")
		f1=open("page.html","w")
		f1.write(str(r.content))
		f1.close()
		f2=open("page.html","r")
		soup = BeautifulSoup(f2.read())
		f2.close()
		#titles=soup.find_all("div", {"class": "a-row"})
		#fp_titles=open("titles.txt","a+")
		#fp_titles.write(str(titles))
		#fp_titles.close()
		review_data = soup.find_all("span", {"class": "a-size-base review-text"})
		date_data = soup.find_all("div", {"class": "a-expander-content reviewText review-text-content a-expander-partial-collapse-content"})
		#print(date_data)
		#time.sleep(250)
		pattern_date='on (.*?)</span>'
		date_a=re.findall(pattern_date,str(date_data))
		all_dates=all_dates+date_a[2:]
		#f3=open("reviews.txt","a+")
		print(str(review_data))
		print("^^^^^^^^^^^^^^^^soup content end^^^^^^^^^^^^^^^^^^^\n")
		#for sen in f3:
		#	if '</div>' not in sen:
		#		f3.write(str(review_data))
		#	else:
		#		f3.write("\n")
		
		pattern='<span class="a-size-base review-text" data-hook="review-body">(.*?)</span>'
		review=re.findall(pattern,str(review_data))
		#f3.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
		#f3.write(str(review))
		#f3.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
		print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
		print(str(review))
		print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
		review_dir={'Review':review}
		review_dir_frame=pd.DataFrame(review_dir)
		print(" DATAFRAME SMALL-->  ")
		print(review_dir_frame)
		Review_Frame=Review_Frame.append(review_dir_frame,ignore_index=True)
	for s in all_dates:
		date_object = datetime.strptime(s , '%d %B %Y')
		date_frame=date_frame.append({'Date':date_object},ignore_index=True)
	print(date_object)
	#time.sleep(500)
	print(" FINAL DATAFRAME -->  ")
	#all=Review_Frame
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
	i=0
	#negcutoff = int(len(negfeats)*0.5)
	#poscutoff = int(len(posfeats)*0.5)
	trainfeats = posfeats[:20]+negfeats[:20]  
	#cl = NaiveBayesClassifier(trainfeats)
	#saveclassifier=open("naivebayes.pickle","wb")
	#pickle.dump(cl,saveclassifier)
	#saveclassifier.close()
	saveclassifier=open("naivebayes.pickle","rb")
	cl=pickle.load(saveclassifier)
	saveclassifier.close()
	REVIEW_WITHOUT_STOPWORDS=[]
	print("TRAINING COMPLETE")
	print("ANALYZING.....")
	val1=[]
	print("\n\n*************date*************\n")
	print(singlelink)
	#print(Review_Frame)
	print(all_dates)
	print("\n\n*************date*************\n")
	#time.sleep(250)
	
	for index,row in Review_Frame.iterrows():
		REVIEW_TOKENIZE=tknzr.tokenize(row['Review'])
		sum=0
		count=-1
		#val1=[]
		for w in REVIEW_TOKENIZE:
			if w not in stop_words and wordnet.synsets(w):
				prob_dist = cl.prob_classify(w)
				print('prob dist=' )
				print(prob_dist.prob("pos"))
				print(prob_dist.prob("neg"))
				positive=round(prob_dist.prob("pos"), 2)
				val=positive*100
				
				count += 1
        #DEPENDS ON TRAINING DATA 
				if val in range(0,30,1):
					#synon = wordnet.synsets(w)
					#if synon:
					val1.insert(count,val*-1)
					Target_Word_Frame=Target_Word_Frame.append({'Target_Word':w.lower(),'Polarity':val,'Tag':'NEGATIVE'},ignore_index=True)
				elif val in range(30,50,1):
					#synon = wordnet.synsets(w)
					#if synon:
					val1.insert(count,val)
					Target_Word_Frame=Target_Word_Frame.append({'Target_Word':w.lower(),'Polarity':val,'Tag':'CONFLICT'},ignore_index=True)
				else:
					#synon = wordnet.synsets(w)
					#if synon:
					val1.insert(count,val*1)
					Target_Word_Frame=Target_Word_Frame.append({'Target_Word':w.lower(),'Polarity':val,'Tag':'POSITIVE'},ignore_index=True)
		average=statistics.mean(val1)
		if average<50.0:
			frame=frame.append({'Review':row['Review'],'Tag':'NON-SATISFACTORY','Rate':average},ignore_index=True)
		else:
			frame=frame.append({'Review':row['Review'],'Tag':'SATISFACTORY','Rate':average},ignore_index=True)
		#review_rate=Review_Frame+all
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
	fontsize = 35
	font = ImageFont.truetype("arial.ttf", fontsize)

	draw.text((10, 0),'\n\n' + "POSITIVE : "+ str(RESULT_POS_N) + '\n'+ '\n'+ "NEGATIVE : "+ str(RESULT_NEG_N) + '\n' + '\n'+ "CONFLICT : "+ str(RESULT_CON_N) , (0,0,0), font = font)
	img_resized = image.resize((188,10), Image.ANTIALIAS)

	image.save("webapp/static/images/amaimg.png")
	
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
	
	
	fig=x_y.plot(x='Date', y='Rate',title="Amazon Monthly Analysis")
	plt.xlabel("Months")
	plt.ylabel("Rate")
	fig1 = plt.gcf()
	plt.show()
	
	fig1.savefig('webapp/static/images/amaMonthly.png')
	Target_Word_Frame.Tag.value_counts().plot(kind='barh')
	plt.title('REVIEW TYPES')
	plt.xlabel('Frequency')
	fig=plt.gcf()
	plt.show()
	fig.savefig('webapp/static/images/amabar.png')
	Target_Word_Frame.Tag.value_counts().plot(kind='pie')
	plt.axis('equal')
	plt.title('Number of appearances in dataset')
	fig=plt.gcf()
	plt.show()
	fig.savefig('webapp/static/images/amapie.png')
	#ax = s.hist()  # s is an instance of Series
	#fig = ax.get_figure()
	#fig.savefig('/path/to/figure.pdf')
	print("-------------------REVIEWS BOXPLOT---------------------------------\n")
	#plt.show(Target_Word_Frame.boxplot(column='Polarity'))
	print("@@@@@@@@@@@@@@@@@@@@@@@END@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
	#review_rate=[]
	#review_rate.append(Review_Frame)
	#review_rate.append(all)
	#print('00000000000000\n')
	#print(review_rate)
	#print('\n00000000000000\n')
	
	#print(frame)
	#new=pd.concat([Review_Frame, frame],axis=1)
	#print(new)
	return frame
	
	
def getlink(product):
	#pd.set_option('display.width',1000)
	pd.options.display.max_colwidth = 200
	site ="amazon.in"  
	#product ="NEXUS 5"
	start="'href': '/url?q="  
	end="&sa" 
	frame = pd.DataFrame()
	links = linkGrabber.Links('https://www.google.com/search?newwindow=1&biw=1366&bih=659&q='+site+'+'+product+'+product+reviews')
	gb = links.find(limit=30)
	print("~~~~~~~~~~~~print g2[25]~~~~~~~~~~~~~~~ ")
	print(gb[25])
	gb1=str(gb)
	print("~~~~~~~~~~~print g2 group(1)~~~~~~~~~~~~~")
	print(str(gb1))
	
	gb2 = re.search("https://www.amazon.com/(.+?)(%|&)",gb1)
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("LINK GRABBED IS: ")
	print(gb2)
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("print g2 group(1)")
	print(gb2.group(1))
	if gb2:
		'''found = gb2.group(1)
		print(found[-10:])
		found1=found+'xyz'
		#time.sleep(4)
		gb3 = re.search('/(.+?)xyz',found1)
		found2=gb3.group(1)
		print(found2)
		time.sleep(500)
		frame =scrap(found2)'''
		found=gb2.group(1)
		print(found[-10:])
		frame =scrap(found[-10:])
	else:
		print('Not found')
	return frame
    
getlink("iphone xs")