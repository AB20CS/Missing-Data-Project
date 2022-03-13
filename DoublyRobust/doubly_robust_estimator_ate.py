### Doubly Robust Model to estimate Average Treatment Effect (ATE)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

def doubly_robust(df, X, T, Y):
    ps = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

if __name__ == "__main__":
    T = 'intervention'
    Y = 'achievement_score'
    data = pd.read_csv("./learning_mindset.csv")
    categ = ["ethnicity", "gender", "school_urbanicity"]
    cont = ["school_mindset", "school_achievement", "school_ethnic_minority", "school_poverty", "school_size"]

    data_with_categ = pd.concat([
        data.drop(columns=categ), # dataset without the categorical features
        pd.get_dummies(data[categ], columns=categ, drop_first=False) # categorical features converted to dummies
    ], axis=1)

    X = data_with_categ.columns.drop(['schoolid', T, Y])

    print(doubly_robust(data_with_categ, X, T, Y))