
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from datetime import date, timedelta
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

class ENCovidDatasetLoader(object):
    """A dataset of mobility and history of reported cases of COVID-19 in England NUTS3 
    regions, from 3 March to 12 of May.
    The dataset is segmented in days and the graph is directed and weighted. 
    The graph indicates how many people moved from one region to the other each day, based on Facebook Data For 
    Good disease prevention maps (https://dataforgood.fb.com/tools/disease-prevention-maps).
    The node features correspond to the number of COVID-19 cases in the region in the past **window** days.
    The task is to predict the number of cases in each node after 1 day.
    For details see this paper: `"Transfer Graph Neural Networks for Pandemic Forecasting
." <https://arxiv.org/abs/2009.08388>`.
    
    Args:
        window (int): Number of past day measurements used for node features.
        scaled (bool): Normalize the features.
    """
    def __init__(self, window: int, scaled: bool=False):
        self.window = window
        self.scaled = scaled
        self.read_web_data()

    def read_web_data(self):
        labels = pd.read_csv("https://github.com/geopanag/pandemic_tgnn/blob/master/data/England/england_labels.csv?raw=true")
        labels = labels.set_index("name")

        sdate = date(2020, 3, 13)
        edate = date(2020, 5, 12)
        delta = edate - sdate
        dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
        dates = [str(date) for date in dates]
        
        labels = labels.loc[:,dates]    
        
        Gs = self.download_graphs(dates,"EN")
        labels = labels.loc[list(Gs[0].nodes()),:]
        
        features = self.generate_features(Gs ,labels ,dates)
        
        gs_adj = [csr_matrix(nx.adjacency_matrix(kgs).toarray().T) for kgs in Gs]
        edge_index = [kgs.indices for kgs in gs_adj]
        edge_weight = [kgs.data for kgs in gs_adj]
        
        y = list()
        for i,G in enumerate(Gs):
            y.append(list())
            for node in G.nodes():
                y[i].append(labels.loc[node,dates[i]])
                
        self._edge_index=edge_index
        self._edge_weight = edge_weight
        self.features = features
        self.targets = y


    def generate_features(self, Gs, labels, dates,scaled = False ):
        """
        Generate the node features based on the numebr of cases in each day.
        Features[0] contains the features to predict y[0].
        e.g. if window = 7, features[7]= day0:day6, y[7] = day7
        if the window goes before 0, everything is 0, so features[3] = [0,0,0,0,day0,day1,day2], y[3] = day3
        """
        features = list()
        
        labs = labels.copy()
        
        for idx, G in enumerate(Gs):
            H = np.zeros([G.number_of_nodes(),self.window])
            me = labs.loc[:, dates[:(idx)]].mean(1)
            sd = labs.loc[:, dates[:(idx)]].std(1)+1
    
            
            for i,node in enumerate(G.nodes()):
            
                if(idx < self.window):
                    if(self.scaled):
                        
                        H[i,(self.window-idx):(self.window)] = (labs.loc[node, dates[0:(idx)]] - me[node])/ sd[node]
                    else:
                        H[i,(self.window-idx):(self.window)] = labs.loc[node, dates[0:(idx)]]
    
                elif idx >= self.window:
                    if(scaled):
                        H[i,0:(self.window)] =  (labs.loc[node, dates[(idx-self.window):(idx)]] - me[node])/ sd[node]
                    else:
                        H[i,0:(self.window)] = labs.loc[node, dates[(idx-self.window):(idx)]]
          
                
            features.append(H)
            
        return features


    def download_graphs(self,dates,country):
        """
        Download the list of graphs of each day.
        """
        Gs = []
        for dat in dates:
            d = pd.read_csv("https://raw.githubusercontent.com/geopanag/pandemic_tgnn/master/data/England/graphs/EN_"+dat+".csv",header=None)
            G = nx.DiGraph()
            nodes = set(d[0].unique()).union(set(d[1].unique()))
            G.add_nodes_from(nodes)
            
            for row in d.iterrows():
                G.add_edge(row[1][0], row[1][1], weight=row[1][2])
                
            Gs.append(G)
            
        return Gs        
    
    
    def get_dataset(self) -> DynamicGraphTemporalSignal:
        """Returning the COVID-19 England data iterator.
        Return types:
            * **dataset** *(DynamicGraphTemporalSignal)* - The COVID19EN dataset.
        """
        dataset = DynamicGraphTemporalSignal(self._edge_index, self._edge_weight, self.features, self.targets)
        return dataset
