import numpy as np
from typing import Sequence, Union
from clustiseq import TimedSequence
from clustiseq.metrics.dtw import drop_dtw
import warnings

"""
* predict function based on the metric
"""


class KMeans:
    def __init__(self, nb_clusters : int = 4, 
                 max_iterations : int = 30,
                 stop_criteria : float = 0.0):
        """
        Parameters
        ----------
        nb_clusters : Int
            Number of clusters requested by the user. Default is 1.

        Note
        ----------
        We suggest to look at the (drop)DTW description before choosing the different parameters.
        """
        self.nb_clusters = nb_clusters
        self.nb_it = max_iterations
        self.sim_threshold = stop_criteria

        self.metric = drop_dtw(tc_param=None, sc_param=None,drop_cost=1,t_weight=1,ev_weight=1)
        self.dtw_itmax = 5 #number of iterations for average

        self.centroids = None

    def fit(self, dataset : Sequence[TimedSequence]):
        """KMeans clustering with Dynamic Time Warping
            
            Parameters
            ----------
            dataset : List
                List timed sequences

            Returns
                * the cluster description
                * the average timed sequences
        """

        n = len(dataset)
        if n<self.nb_clusters:
            warnings.warn("too much clusters")
            return [[i] for i in range(self.nb_clusters)], [ dataset[i] for i in range(self.nb_clusters)]

        # copy of the dataset (internally modified)
        # create a uniform vectorial representation of events
        vmax = max([np.max(s._data) for s in dataset])
        for s in dataset:
            s._pdata = np.zeros( (s._data.size, vmax + 1) )
            s._pdata[np.arange(s._data.size), s._data] = 1

        #select randomly the first centroids
        self.centroids = []
        for _ in range(self.nb_clusters):
            self.centroids.append( dataset[np.random.randint(0,n)] )

        for _ in range(self.nb_it):
            assignments=[[] for _ in range(self.nb_clusters)]
            # assign examples to centroids
            for i in range(n):
                vals = np.argmin([ self.metric(self.centroids[k], dataset[i]) for k in range(self.nb_clusters)])
                assignments[ vals ].append(i)

            old_centroids = self.centroids.copy()
            diff=0
            #update centroids
            for k in range(self.nb_clusters):
                if len(assignments[k])>0:
                    self.centroids[k]=self.metric.average( [dataset[i] for i in assignments[k]], itmax=self.dtw_itmax, preparesequences=False )
                else:
                    self.centroids[k]=dataset[np.random.randint(0,n)]
                diff += self.metric(old_centroids[k], self.centroids[k])
            if diff<self.nb_clusters*self.sim_threshold:
                return

            
            

        assignments=[[] for _ in range(self.nb_clusters)]
        for i in range(n):
            assignments[ np.argmin([ self.metric(self.centroids[k], dataset[i]) for k in range(self.nb_clusters)]) ].append(i)
        
        return assignments, self.centroids

    def __predict_one(self, sequence : TimedSequence) -> int:
        return np.argmin([ self.metric(self.centroids[k], sequence) for k in range(self.nb_clusters)])
    
    def predict(self, sequences : Union[TimedSequence,Sequence[TimedSequence]]):
        if isinstance(sequences, list ):
            return [self.__predict_one(seq) for seq in sequences]
        else:
            return self.__predict_one(sequences)



class hierarchical:
    def __init__(self, nb_clusters : int = 4):
        """
        Parameters
        ----------
        nb_clusters : Int
            Number of cluster requested by the user. Default is 1.

        Note
        ----------
        We suggest to look at the DTW description before choosing the different parameters.
        """
        self.nb_clusters = nb_clusters

        self.metric = drop_dtw(tc_param=None, sc_param=None,drop_cost=1,t_weight=1,ev_weight=1)
        self.dtw_itmax = 5 #number of iterations for average

        self.barycenters = None


    def __sim_matrix_from_dataset(self, dataset : Sequence[TimedSequence]):
        """Similarity Matrix From a Dataset

        Parameters
        ----------
        dataset : 
            List timed sequences

        return:
            Return a similarity matrix from the dataset.
        """
        size_ = len(dataset)
        dtw_matrix = np.ndarray((size_,size_),dtype=float) # initialize the similarity matrix
        for i in range(size_):
            dtw_matrix[i][i] = float('inf')
            for j in range(i): # fill the rest of the matrix with the similarity measure between two different elements
                dtw_matrix[i][j] = self.metric(dataset[i],dataset[j])
                dtw_matrix[j][i] = self.metric(dataset[j],dataset[i])

        return dtw_matrix
    
    def fit(self, _dataset : Sequence[TimedSequence]):
        """Hierarchical clustering with Dynamic Time Warping

            Parameters
            ----------
            dataset : List
                List timed sequences

            Returns
                * the cluster description
                * the average timed sequences
        """

        # copy of the dataset (internally modified)
        # create a uniform vectorial representation of events
        vmax = max([np.max(s._data) for s in _dataset])
        dataset=[]
        for s in _dataset:
            ns = TimedSequence(s._dates, s._data)
            ns._pdata = np.zeros( (s._data.size, vmax + 1) )
            ns._pdata[np.arange(s._data.size), s._data] = 1
            dataset.append(ns)

        n = len(dataset)
        matrix_sim = self.__sim_matrix_from_dataset(dataset) # compute the similarity matrix from the dataset

        clusters=[ [i] for i in range(n) ]

        for _ in range(n-self.nb_clusters):
            # compute the indexes of the minimum non-diagonal element of the similarity matrix
            [x,y] = np.argwhere(matrix_sim==np.min(matrix_sim))[0]

            ## Compute the average sequence (_pdata have been precomputed)
            avseq=self.metric.average( [dataset[x],dataset[y]], itmax=self.dtw_itmax, preparesequences= False )

            ## Update the data structures
            #replace the x-th row and column by the new values
            for j in range(n):
                if (j == x):
                    matrix_sim[x][j]=matrix_sim[j][x]=float('inf')
                elif j==y:
                    continue
                else:
                    matrix_sim[x][j]=self.metric(avseq, dataset[j])
                    matrix_sim[j][x]=self.metric(dataset[j], avseq)
            
            #remove the y-th rows and columns of the similarity matrix
            matrix_sim=np.delete(matrix_sim, y, 0)
            matrix_sim=np.delete(matrix_sim, y, 1)

            #replace the x-th sequence by the average
            dataset[x]= avseq
            #remove the y-th sequence
            del dataset[y]

            # gather the classes
            clusters[x] = clusters[x]+clusters[y]
            del clusters[y]

            # next step preparation
            n = len(matrix_sim) # n become the size of the new matrix for the next step

        self.barycenters = dataset
        return clusters, dataset

    def __predict_one(self, sequence : TimedSequence) -> int:
        return np.argmin([ self.metric(self.barycenters[k], sequence) for k in range(self.nb_clusters)])
    
    def predict(self, sequences : Union[TimedSequence,Sequence[TimedSequence]]):
        if isinstance(sequences, list ):
            return [self.__predict_one(seq) for seq in sequences]
        else:
            return self.__predict_one(sequences)