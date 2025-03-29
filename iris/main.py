import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("iris/iris.csv")
    
    xcols = [
      "sepal_length", 
      "sepal_width", 
      "petal_length", 
      "petal_width"
    ]
    ycols = [
      "species"
    ]
    for xcol in xcols:
      # print(col)
      # print(df[col].mean())
      # print(df[col].median())
      df[xcol] = df[xcol].fillna(df[xcol].mean())
    
    xtrain, xtest, ytrain, ytest = train_test_split(df[xcols], df[ycols], test_size=0.3)
    model = tree.DecisionTreeClassifier(max_depth=2)
    model.fit(xtrain, ytrain)
    print(model.score(xtest, ytest))
    
if __name__ == "__main__":
    main()