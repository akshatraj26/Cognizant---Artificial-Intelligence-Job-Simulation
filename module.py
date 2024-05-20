# Importing packages
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler



# Load the data
def load_data(
        path: str=None
        ):
    """
    This take a path of the csv file and load it in the form of pandas dataframe

    Parameters
    ----------
    path : str, optional
        DESCRIPTION.

    Returns   df:pd.DataFrame
    -------
    None.

    """
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df


# Create a dependent and independent variable
def dependent_and_independent_variable(
        data: pd.DataFrame = None, 
        target: str = 'estimated_stock_pct'
        ):
    """
    This will take Pandas dataframe and seprate the dependent and independent variable, i.e X, y.
    this two will be used to train the model using supervised machine learning algorithm

    Parameters
    ----------
    data : pd.DataFrame, optional
        DESCRIPTION. The default is None.
    target : str, optional
        DESCRIPTION. The default is 'estimated_stock_pct'.

    Returns  X: pd.DataFrame
             y: pd.Series
    -------
    None.

    """
    # check if target variable is present or not
    if target not in df.columns:
        raise Exception(f"Target:  {target} is not present in the data")
    
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# Train algorithm with cross validation
def train_algorithm_with_cross_validation(
        X: pd.DataFrame = None,
        y: pd.Series = None
    ):
    """
    This function takes dependent and independent variable and split the data into train and test
    and then scale the independent variable and then also use cross validation and print the accuracy in each fold and return
    accuracy

    Parameters
    ----------
    X : pd.DataFrame, optional
        DESCRIPTION. The default is None.
    y : pd.Series, optional
        DESCRIPTION. The default is None.

    Returns 
    -------
    None.

    """
    
    # create a list that will store the accuracy of each fold
    accuracy = []
    # number of folds
    K = 10
    # enter the fo loop for k=10 fold
    for i in range(0, K):
            
        # instance of the RandomForest
        rf = RandomForestRegressor()
        # intance of the Scaler
        scaler = StandardScaler()
        
        # training and testing samples
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)
        
        # scale the X data so that our model won't be greedy with large value
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        rf.fit(X_train, y_train)
        yhat = rf.predict(X_test)
        ac = mean_absolute_error(y_test, yhat)
        accuracy.append(ac)
        
        print(f"Fold {i+1}: MAE:- {ac:.3f}")
    # finish by computing the average of the accuracy
    print(f"Average MAE:- {(sum(accuracy) / len(accuracy)):.3f}")
    
    
# Now i will combine everything together in modular fundction
def main():
    """
    This function takes the file path and return pandas dataframe and then sepeate the predictor and target variable
    and then split the data into train and test samples. after that start training the model

    Returns : float value
    -------
    None.

    """
    # load the data
    df = load_data()
    
    # split the data into a target and predictor
    X, y = dependent_and_independent_variable(df=df)
    
    # train the model
    train_algorithm_with_cross_validation(X=X, y=y)