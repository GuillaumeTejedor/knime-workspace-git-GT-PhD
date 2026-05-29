import os

from snorkel import SnorkelSession
from snorkel.parser import TSVDocPreprocessor

from snorkel.parser.spacy_parser import Spacy
from snorkel.parser.rule_parser import RuleBasedParser,RegexTokenizer
from snorkel.parser import CorpusParser

from snorkel.models import Document, Sentence
from snorkel.models import candidate_subclass, TemporarySpan

from snorkel.candidates import Ngrams, CandidateExtractor, CandidateSpace
from snorkel.matchers import RegexMatch,PersonMatcher, RegexMatchEach,RegexMatchSpan

from snorkel.contrib.brat import BratAnnotator

import re
from snorkel.lf_helpers import (
    get_left_tokens, get_right_tokens, get_between_tokens,
    get_text_between, get_tagged_text,
)



from snorkel.annotations import LabelAnnotator

import numpy as np
from numpy import dot
from numpy.linalg import norm

from snorkel.learning import GenerativeModel
import matplotlib.pyplot as plt

import csv 



class queryCandidate(CandidateSpace):
    """
    Defines the space of candidates as pairs of consecutive queries.
    """
    def __init__(self):
        CandidateSpace.__init__(self)
    
    def apply(self, context):
 #       print(context.position)
        seen = set()
        text=context.text # gets sentence as string
        #print(text)
        i=0
        while i < len(text)-1:
            j=i+1
            while text[j]!=';':
                j=j+1
            # must continue until next one
            k=j+1
            j=j+1
            if j<len(text)-1:
                #print(j,len(text))
                while text[j]!=';':
                    j=j+1    
                start=i
                end=j
                #i=j+1
                i=k
           
                #print(start,end)
                #print(text[start:end])
                ts    = TemporarySpan(char_start=start, char_end=end, sentence=context)
                if ts not in seen:
                    seen.add(ts)
                    yield ts
            else:
                i=j





# one big document with one sentence per exploration
# if -small -> just a little sample for testing purpose
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshareQueries-IDs-small.txt' 
path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/smartBIqueries-IDs.txt' 

session = SnorkelSession()

doc_preprocessor = TSVDocPreprocessor(path)

corpus_parser = CorpusParser(parser=Spacy())

corpus_parser.apply(doc_preprocessor)
print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())
#thedoc=session.query(Document).all()[0]



pairs = candidate_subclass('pairs1', ['queryPair'])
regexpmatch=RegexMatchSpan(rgx=".*")
cs=queryCandidate()
cand_extractor = CandidateExtractor(pairs, [cs], [regexpmatch])


docs = session.query(Document).order_by(Document.name).all()
sentences = session.query(Sentence).all()
#print(sentences)

sents=set();
for i,doc in enumerate(docs):
    for s in doc.sentences:
        sents.add(s)


cand_extractor.apply(sents)
    
print("Number of candidates:", session.query(pairs).count())

def printpairs(pairs):
    for cand in session.query(pairs):
        print('--------------------------------------------------')
        print(cand.queryPair)
        print('--------------------------------------------------')

#printpairs(pairs)

# 1=true, -1=false, 0=no label


#metricspath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/smartBI-legros-v1.csv'
metricspath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/smartBI-legros-v1-nolength.csv'
# in that file, we have sessionID first and then queryId
# while candidates are queryID,sessionID

#fieldNames="QuerySId,SessionSId,UserSId,NoP,NoF,NoA,NoT,NoAtt,NCP,NCF,NCA,NCT,NCAtt,zNoP,zNoS,zNoA,zNoT,zNoAtt,zNCP,zNCF,zNCA,zNCT,zNCAtt,NoQ,Lenght,RED,Edit-index,Jackard-index,Cosine-index,Common-fragments-index,Common-tables-index,Vote,ExplorationSId,GroundTruth,ChangeSession"

fieldNames="QuerySId,SessionSId,UserSId,NoP,NoF,NoA,NoT,NoAtt,NCP,NCF,NCA,NCT,NCAtt,zNoP,zNoS,zNoA,zNoT,zNoAtt,zNCP,zNCF,zNCA,zNCT,zNCAtt,NoQ,RED,EditIndex,JackardIndex,CosineIndex,CommonFragmentsIndex,CommonTablesIndex,Vote,ExplorationSId,GroundTruth,ChangeSession"



metrics=np.genfromtxt(metricspath, dtype=None, delimiter=';', names=fieldNames, encoding='utf8', skip_header=1)

def recall(name,idQuery1,idQuery2,idSession):
    num=findMetric(name,idQuery2,idSession)
    metric2=name.replace('C','o')
    denom=findMetric(metric2,idQuery1,idSession)
    if denom!=0:
        return num/denom
    else:
        return 0
    
def precision(name,idQuery,idSession):
    num=findMetric(name,idQuery,idSession)
    metric2=name.replace('C','o')
    denom=findMetric(metric2,idQuery,idSession)
    if denom!=0:
        return num/denom
    else:
        return 0

def lessInCommon(name,idQuery1,idQuery2,idSession):
    numQ1=findMetric(name,idQuery1,idSession)
    numQ2=findMetric(name,idQuery2,idSession)
    if numQ2<=numQ1:
        return True
    else:
        return False

def findMetric(name,idQuery,idSession):
    if idQuery==0:
        return 0
    else:
        tmp=metrics[metrics['SessionSId']==idSession]
        return tmp[tmp['QuerySId']==idQuery][name][0]
    

    
def getCandidatesIDs(c):
    pair=c.queryPair.get_span()
    q1=pair.split(sep=';')[0]
    q2=pair.split(sep=';')[1]
    idQ1=q1.split(sep=',')[0]
    idSession=q1.split(',')[1]
    idQ2=q2.split(sep=',')[0]
    return (int(idQ1),int(idQ2),int(idSession))
    
    
    
    
    
# LF functions 
# 1: together
# -1: split
# 0: otherwise
def LF_recall_projections(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCP',idQ1,idQ2,idSession)
    if rec==1: 
        return 1
    elif rec==0:
        return -1
    else:
        return 0
        

def LF_precision_projections(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCP',idQ2,idSession)
    if prec==1: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_selections(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCF',idQ1,idQ2,idSession)
    if rec==1: 
        return 1
    elif rec==0:
        return -1
    else:
        return 0
        

def LF_precision_selections(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCF',idQ2,idSession)
    if prec==1: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_aggregation(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCA',idQ1,idQ2,idSession)
    if rec==1: 
        return 1
    elif rec==0:
        return -1
    else:
        return 0
        

def LF_precision_aggregation(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCA',idQ2,idSession)
    if prec==1: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_tables(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCT',idQ1,idQ2,idSession)
    if rec==1: 
        return 1
    elif rec==0:
        return -1
    else:
        return 0
        

def LF_precision_tables(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCT',idQ2,idSession)
    if prec==1: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        

## version 2
##
## favors keeping queries together 
##
def LF_recall_projections2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCP',idQ1,idQ2,idSession)
    if rec!=0: 
        return 1
    else:
        score=findMetric('NoP',idQ1,idSession)
        if score==0:
            return 0
        else:
            return -1
        

def LF_precision_projections2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCP',idQ2,idSession)
    if prec!=0: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_selections2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCF',idQ1,idQ2,idSession)
    if rec!=0: 
        return 1
    else:
        score=findMetric('NoF',idQ1,idSession)
        if score==0:
            return 0
        else:
            return -1
        

def LF_precision_selections2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCF',idQ2,idSession)
    if prec!=0: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_aggregation2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCA',idQ1,idQ2,idSession)
    if rec!=0: 
        return 1
    else:
        score=findMetric('NoA',idQ1,idSession)
        if score==0:
            return 0
        else:
            return -1
        

def LF_precision_aggregation2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCA',idQ2,idSession)
    if prec!=0: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_tables2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCT',idQ1,idQ2,idSession)
    if rec!=0: 
        return 1
    else:
        score=findMetric('NoT',idQ1,idSession)
        if score==0:
            return 0
        else:
            return -1
        

def LF_precision_tables2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCT',idQ2,idSession)
    if prec!=0: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
##
## less in common

def LF_lessInCommonProjection(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    less=lessInCommon('NCP',idQ1,idQ2,idSession)
    if less==True:
        return -1
    else:
        return 1
      
def LF_lessInCommonSelection(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    less=lessInCommon('NCF',idQ1,idQ2,idSession)
    if less==True:
        return -1
    else:
        return 1
        
def LF_lessInCommonAggregation(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    less=lessInCommon('NCA',idQ1,idQ2,idSession)
    if less==True:
        return -1
    else:
        return 1
  
def LF_lessInCommonTable(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    less=lessInCommon('NCT',idQ1,idQ2,idSession)
    if less==True:
        return -1
    else:
        return 1
          
        


# indexes as defined in DOLAP 2019 paper:
#


def LF_edit_index(c):
    (idQ1, idQ2, idSession) = getCandidatesIDs(c)
    edit_index = findMetric('EditIndex', idQ2, idSession)
    if edit_index > 0:
        return 1
    else:
        return -1


def LF_jackard_index(c):
    (idQ1, idQ2, idSession) = getCandidatesIDs(c)
    jackard_index = findMetric('JackardIndex', idQ2, idSession)
    if jackard_index > 0:
        return 1
    else:
        return -1


def LF_cosine_index(c):
    (idQ1, idQ2, idSession) = getCandidatesIDs(c)
    cosine_index = findMetric('CosineIndex', idQ2, idSession)
    if cosine_index > 0:
        return 1
    else:
        return -1


def LF_common_fragment_index(c):
    (idQ1, idQ2, idSession) = getCandidatesIDs(c)
    common_fragment_index = findMetric('CommonFragmentsIndex', idQ2, idSession)
    if common_fragment_index > 0:
        return 1
    else:
        return -1


def LF_Common_Tables_Index(c):
    (idQ1, idQ2, idSession) = getCandidatesIDs(c)
    common_tables_index = findMetric('CommonTablesIndex', idQ2, idSession)
    if common_tables_index > 0:
        return 1
    else:
        return -1


# testing   
def testLFs(): 
    for c in session.query(pairs):
        print(c.queryPair.sentence.text[c.queryPair.char_start:c.queryPair.char_end])
        print(c.labels)



#LFs=[LF_recall_projections,LF_precision_projections,LF_recall_selections,LF_precision_selections,LF_recall_aggregation,LF_precision_aggregation,LF_recall_tables,LF_precision_tables]
#LFs=[LF_recall_projections2,LF_precision_projections2,LF_recall_selections2,LF_precision_selections2,LF_recall_aggregation2,LF_precision_aggregation2,LF_recall_tables2,LF_precision_tables2] # best score so far
#LFs=[LF_lessInCommonProjection,LF_lessInCommonSelection,LF_lessInCommonAggregation,LF_lessInCommonTable]
#LFs=[LF_recall_tables2]
#LFs=[LF_recall_aggregation,LF_recall_tables,LF_recall_projections2,LF_recall_selections2,LF_recall_tables2,LF_precision_tables2]
#LFs=[LF_edit_index,LF_jackard_index,LF_cosine_index,LF_common_fragment_index,LF_Common_Tables_Index]
LFs=[LF_jackard_index]




# def get_power_set(s):
#   power_set=[[]]
#   for elem in s:
#     # iterate over the sub sets so far
#     for sub_set in power_set:
#       # add a new subset consisting of the subset at hand added elem
#       power_set=power_set+[list(sub_set)+[elem]]
#   return power_set
#   
# posetLF=get_power_set(LFs)
# posetLF=posetLF[1:-1]

#for LFs in posetLF:

labeler = LabelAnnotator(lfs=LFs)

#np.random.seed(1701)
L_train = labeler.apply()
#print(L_train)
print(L_train.lf_stats(session))


# generative model, training_marginals are probabilistic training labels
gen_model = GenerativeModel()
gen_model.train(L_train, epochs=100, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=1e-6)


print(gen_model.weights.lf_accuracy)

train_marginals = gen_model.marginals(L_train)

plt.hist(train_marginals, bins=20)
plt.show()

print(gen_model.learned_lf_stats())




# check ground truth


matchpath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/check_match.csv'
with open(matchpath, mode='w') as match_file:
    match_writer = csv.writer(match_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    match_writer.writerow(['idSession','idQuery','marginal','cut'])
    i=0
    tp=0
    tn=0
    fp=0
    fn=0
    for c in session.query(pairs):
        #print('i=',i)
        #print('marginal=',train_marginals[i])
        (idQ1,idQ2,idSession)=getCandidatesIDs(c)
        #print('session=',idSession)
        #print('idQ1=',idQ1)
        #print('idQ2=',idQ2)
        #print('--------')    
        tmp=metrics[metrics['SessionSId']==idSession]
        cut=tmp[tmp['QuerySId']==idQ2]['GroundTruth'][0]
        session=tmp[tmp['QuerySId']==idQ2]['SessionSId'][0]
        query=tmp[tmp['QuerySId']==idQ2]['QuerySId'][0]
        marginal=train_marginals[i]
        if cut==1 and marginal<0.8:
            tp=tp+1
        if cut==0 and marginal>=0.8:
            tn=tn+1
        if cut==0 and marginal<0.8:
            fp=fp+1
        if cut==1 and marginal>=0.8:
            fn=fn+1
        #print('session=',session)
        #print('query order=',query)
        #print(' cut=',1-cut)
        match_writer.writerow([idSession,idQ2,marginal,cut])
        i=i+1
    wprecision=tp/(tp+fp)
    wrecall=tp/(tp+fn)
    waccuracy=(tp+tn)/(tp+fn+fp+tn)
    wfmeasure=(2*wprecision*wrecall)/(wprecision+wrecall)
    print(LFs)    
    print('F-measure=',wfmeasure)
    print('precision=',wprecision)
    print('recall=',wrecall)
    print('accuracy=',waccuracy)

 

#L_dev = labeler.apply_existing()
