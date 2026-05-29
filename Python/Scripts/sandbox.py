import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn import svm
import os, re, sys
import pathlib

path_csv = str(pathlib.Path(__file__).parent.resolve()) + "\\..\\..\\IO Files\\Stratification\\ft_labeled_pairs_sample.csv"

df = pd.read_csv(path_csv)
X = np.array(df[["DURATION_DIFF", "FV_DIFF"]])
y = np.array(df.iloc[:, -1])

print(X)
print(y)

plot_offset = 1000

def linear_svm(X, y):
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    w = clf.coef_[0]
    a = -w[0] / w[1]
    b = - (clf.intercept_[0]) / w[1]
    xx = np.linspace(min(X[:, 0]) - plot_offset, max(X[:, 0]) + plot_offset)
    yy = a * xx + b
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add scatter plot for class 1
    fig.add_trace(go.Scatter(
        x=X[y == 1, 0], y=X[y == 1, 1],
        mode='markers',
        marker=dict(color='blue', symbol='circle', line=dict(color='black', width=1)),
        name='Class 1'
    ))
    
    # Add scatter plot for class 0
    fig.add_trace(go.Scatter(
        x=X[y == 0, 0], y=X[y == 0, 1],
        mode='markers',
        marker=dict(color='red', symbol='circle', line=dict(color='black', width=1)),
        name='Class 0'
    ))
    
    # Add support vectors
    fig.add_trace(go.Scatter(
        x=clf.support_vectors_[:, 0], y=clf.support_vectors_[:, 1],
        mode='markers',
        marker=dict(size=10, color='black', symbol='circle', opacity=0.5),
        name='Support Vectors'
    ))
    
    # Add decision boundary
    fig.add_trace(go.Scatter(
        x=xx, y=yy,
        mode='lines',
        line=dict(color='black', dash='solid'),
        name='Decision Boundary'
    ))

    # Print details
    print("Weights =", clf.coef_)
    print("Intercept =", clf.intercept_)
    print("b =", b)
    print("a =", a)
    print("Measure =", X[:, 0] * w[0] + X[:, 1] * w[1])
    
    fig.update_layout(
        title='Linear SVM with Decision Boundary',
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        showlegend=True
    )
    
    fig.show()

linear_svm(X, y)