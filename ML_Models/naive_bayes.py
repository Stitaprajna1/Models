import pandas as pd
import numpy as np

class NaiveBayes:

    def __init__(self):
        self.columns_probability_distribution = {}
        self.columns = None
        self.no_of_classes = None
        self.class_labels = []
        self.class_probabilities = []

    def gaussain_probability(self, mu, sigma, x):
        term1 = 1/(2*np.pi*sigma**2)**0.5
        term2 = np.exp(-(x - mu)**2/sigma)
        p = term1*term2
        return p

    def fit(self, X, y, solver = 'gaussian'):
        n = len(X)
        self.columns = X.columns
        self.class_labels = y.unique()
        self.no_of_classes = len(self.class_labels)
        
        for column in X.columns:
            if X[column].dtype == 'object':
                prob_dis = pd.crosstab(X[column], y)/n
                self.columns_probability_distribution[column] = {
                    'type': 'object',
                    'column_name': column,
                    'distribution': prob_dis.reset_index()
                }
            else:
                mean = np.mean(X[column])
                std = np.std(X[column])
                self.columns_probability_distribution[column] = {
                    'type':'number',
                    'column_name': column ,
                    'distribution': {{'mean': np.mean(X.loc[y[y==c],:][column]),
                                     'std': std
                                     } for c in self.class_labels}
                }
                

    def predict(self, x):
        # intialize
        probabilities = [0 for _ in range(self.no_of_classes)]

        for c in self.class_labels:
            joint_class_probability = 1
            class_id = np.where(self.class_labels == c)[0][0]
            for _, pc in  self.columns_probability_distribution.items():
                # unpack values
                col_name = pc['column_name']
                val = x[col_name].values[0]
                df = pc['distribution']
                # calculate joint probability distribution
                if pc['type'] == 'object':
                    p = df[df[col_name]==val][c].values[0]
                    joint_class_probability = joint_class_probability*p
                else:
                    mu = df[c]['mean']
                    sigma = df[c]['sigma']
                    p = self.gaussain_probability(mu=mu, sigma=sigma, x=val)
                    joint_class_probability = joint_class_probability*p

            probabilities[class_id] = joint_class_probability
        
        res_id = np.argmax(probabilities)
        return self.class_labels[res_id]

# if __name__ == '__main__':
#     df = pd.read_csv('Models/play_tennis.csv')

#     # train
#     nv = NaiveBayes()
#     X = df.iloc[:, :-1]
#     y = df['play']
#     nv.fit(X,y)

#     # test
#     print(nv.predict(df[df.index==8].iloc[:, :-1]))