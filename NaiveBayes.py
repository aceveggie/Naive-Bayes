from __future__ import division


from numpy import *
from scipy import *

import math
import random
import os
import operator

def train(data,labs):
    data = array(data)
    labs = array(labs)
    u_labs = []
    u_labs_i=[]
    assert(len(data) == len(labs))

    # get number of features
    no_of_features = len(data[0])

    for i in data:
        assert(no_of_features == len(i))
      


    ############ collect useful information about data
    #get unique labels

    uniq = {}
    prior_probs = {}
    for eachitem in labs:
        uniq[eachitem] = 0
        prior_probs[eachitem] = 0

    for eachitem in uniq.iterkeys():
        u_labs.append(eachitem)

    for eachitem in u_labs:
            c = where((labs==eachitem) )

            prior_probs[eachitem] = len(c[0])/len(data)

            u_labs_i.append(where (labs==eachitem) )


    node = {}

    for eachitem in zip(u_labs_i,u_labs):
        gen = None
        m = data[eachitem[0]]
        s = m.shape[1]
        p = []
        d_array = []
        for i in range(s):
            mm = m[:,i]
            mean_mm = mean(mm)
            var_mm = var(mm,axis=0,ddof=1)
            
            d_array.append([mean_mm, var_mm])
        
        node[eachitem[1]]=d_array

    

    return node, prior_probs


def predict(nodes, prior_probs, pred_data_list):

    pred_output=[]
    for pred_data in pred_data_list:
        
        #calculate posterior probabilities
        post={}
        for eachnode in nodes.iteritems():
            f = eachnode[1]
            post[eachnode[0]]= prior_probs[eachnode[0]]
            ii=0
            for eachitem in f:
                
                #print eachitem,ii,'--'
                mean_m = eachitem[0]
                var_m = eachitem[1]
                
                data_m = pred_data[ii]
                
                
                p1 = 1.00 /(sqrt(2 * pi * var_m))
                p2 = exp ( ( -1 * pow( ( data_m - mean_m ),2 ) ) / (2.00 * var_m))
                post[eachnode[0]] *= (p1 * p2)

                
                ii += 1
                pass
            pass
        pass

        # calculate evidence

        evidence = 0
        
        for eachnode in post.iteritems():
            evidence += eachnode[1]
        pass

        for i in post.iterkeys():
            post[i] = post[i]/evidence
        pass
        
        
        likelilabel = max(post.iteritems(), key=operator.itemgetter(1))[0]

        likelihood = max(post.iteritems(), key=operator.itemgetter(1))[1]
        pred_output.append((likelilabel, likelihood))

    return pred_output
