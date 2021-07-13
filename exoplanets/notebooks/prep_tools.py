# -------------------------------------------- #
# Funciones para cargar , e inspeccionar datos # 
# -------------------------------------------- #
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load(from_url=True,loader=None,target_name='bla'):
    if from_url:
        # user requested loading from url
        dataset = pd.read_csv(url)
    else:
        if loader is None:
            raise Exception('Must provide a loader!')
        # user requeste loading using loader
        dataset, target = loader
        dataset[target_name] = pd.Series(target, index=dataset.index)
         
    return dataset
    
def inspect(df,n_atributes=9,stats=True,distro=True,target_name=None):
    # this assumes target is the last column and the rest are attributes
    attributes = df.columns[:-1].values
    if target_name is None:
        target_name = df.columns[-1]
    
    target_values = np.unique(df[target_name].values)
    print(' Attributes : %d \n%s ' %(len(attributes),attributes))
    print('')
    print(' Target : %s' %target_name)
    print('')
    print(' Target possible values: %s' %target_values)
    print('')
    print('Checking for NaNs:')
    for attr in attributes:
        print('Attribute %30s has NaNs : %s' %(attr, df[attr].isnull().values.any()))

    if stats:
        # check basic stats
        print('\n Basic Statistics (before preprocess) \n')
        print(df.describe())
    
    if distro:
        print('\n Example data distributions : \n')
        # check attirbute value distributions
        import matplotlib.pyplot as plt
        f , ax =plt.subplots(figsize=(6,6))
        for i,attr in enumerate(attributes[:n_atributes]):
            ax = plt.subplot(3,3,i+1)
            df[attr].hist(bins=20,ax=ax)
            ax.set_title(attr)
            ax.grid(False)
        plt.tight_layout()
    
    return attributes,target_name

def preprocess(data,discretize=False,n_discrete_bins=2):
    # Standarize data , zero mean and unit variance
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data.astype(int)

def evaluate(label_real, label_pred, nClasses):
    mat = np.array([[np.sum(np.logical_and(label_real==i, label_pred==j)) 
                     for j in np.arange(nClasses)] 
                    for i in np.arange(nClasses)])
    return(mat)

# Defino un decorador para medir el tiempo de ejecucion de una funcion
import time
def count_time(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        total = (time.time() - start) # millisec
        print('funcion %s tiempo de ejecucion : %f' %(str(function),total))
        return result
    return wrapper

# ----------------------------------  # 
# Funcion para aplicacion de diversos #
# metodos de seleccion de atributos   #
# ----------------------------------  # 
def select_attributes(X,y,attributes=[],selectors=None,sname='RecursiveFeatureElimination'):
    
    # dictionary to keep the ids of the selected features
    # ids range [0,n]
    selected_features={}
    for name,selector in selectors.items():  
        print('')
        print('applying features selector : %s' %name)
        print('')
        if name not in selected_features:
            selected_features[name]=[]
        relevant_attributes=[]
        
        fit = selector.fit(X, y)
        
        if name == 'RecursiveFeatureElimination':
            
            selected_features[name] = np.where(fit.ranking_==1.0)[0]
            print('Features Ranking  , ',fit.ranking_)
            print('Features Selected , ',selected_features[name])
            for i in selected_features[name]:
                relevant_attributes.append(attributes[i-1])
                print('Feature %d : is %s' %(i,attributes[i-1]))

        elif name == 'TreesClassifier':
            
            feature_importances = fit.feature_importances_
            ranked = np.argsort(feature_importances)
            # sort in reverse (from highest to lowest)
            selected_features[name] = ranked[::-1][:3]
            print('Features Importance, ',feature_importances )
            print('Features Selected  , ',selected_features[name])
            for i in selected_features[name]:
                relevant_attributes.append(attributes[i-1])
                print('Feature %d : is %s' %(i,attributes[i-1]))
        
        elif name == 'SelectKBest':
            
            scores = fit.scores_
            ranked = np.argsort(scores)
            # sort in reverse (from highest to lowest)
            selected_features[name] = ranked[::-1][:3]
            print('Features Scores   , ', scores)
            print('Features Selected , ',selected_features[name])
            for i in selected_features[name]:
                relevant_attributes.append(attributes[i-1])
                print('Feature %d : is %s' %(i,attributes[i-1]))

        else:
            raise Exception('Not defined features selector!')

    return relevant_attributes

