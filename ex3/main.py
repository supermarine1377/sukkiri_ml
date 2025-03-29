import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def remove_outliers(df, columns, threshold):
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - threshold * std) & (df[col] <= mean + threshold * std)]
    return df

def main():
    df = pd.read_csv("ex3/ex3.csv")

    xcols = ["x1", "x2", "x3"]
    ycol = "target"

    # 外れ値を削除
    df = remove_outliers(df = df, columns = xcols + [ycol], threshold=2)
    
    x_train, x_test, y_train, y_test = train_test_split(df[xcols], df[ycol], test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))

if __name__ == "__main__":
    main()
