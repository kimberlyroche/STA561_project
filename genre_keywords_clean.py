# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:03:14 2019

@author: harsh
"""

import os
import pandas as pd
import csv
import re


#check directory for execution on non-Windows machines.

directory="genres_keywords"

 

output=open("movies_with_keywords.txt",'w')


for file in os.listdir(directory):
   with open(directory+ "\\" +file,'r') as tsv:

   	i=0
   	for line in csv.reader(tsv):
   		i=i+1

   		res=re.search("^genre\t(.*)$", line[0])

   		if(res!=None):
   		   	genres = res.group(1).split("\t")


   		res2=re.search("^title\t(.*)$",line[0])
   		if(res2!=None):
   		   	title = res2.group(1)

   		if(i%7==0):
   			str1=''.join(line)
   			overview=str1.split("\t")[1]
    
    

   print(title,"\t",overview,"\t",genres,file=output,flush=True)
   
output.close()

data=pd.read_csv("movies_with_keywords.txt",sep="\t",header=None,encoding='latin-1')