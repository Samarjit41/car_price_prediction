{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "86fd01eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd # data preprocessing\n",
    "import numpy as np #mathematical computation\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split # to split data into tain and split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.linear_model import Ridge,Lasso\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5bc5983e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=================================loading the data sheet and preprocessing===============================\n",
      "(4340, 8)\n",
      "                       name  year  selling_price  km_driven    fuel  \\\n",
      "0             Maruti 800 AC  2007          60000      70000  Petrol   \n",
      "1  Maruti Wagon R LXI Minor  2007         135000      50000  Petrol   \n",
      "2      Hyundai Verna 1.6 SX  2012         600000     100000  Diesel   \n",
      "3    Datsun RediGO T Option  2017         250000      46000  Petrol   \n",
      "4     Honda Amaze VX i-DTEC  2014         450000     141000  Diesel   \n",
      "\n",
      "  seller_type transmission         owner  \n",
      "0  Individual       Manual   First Owner  \n",
      "1  Individual       Manual   First Owner  \n",
      "2  Individual       Manual   First Owner  \n",
      "3  Individual       Manual   First Owner  \n",
      "4  Individual       Manual  Second Owner  \n",
      "Index(['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type',\n",
      "       'transmission', 'owner'],\n",
      "      dtype='object')\n",
      "name             0\n",
      "year             0\n",
      "selling_price    0\n",
      "km_driven        0\n",
      "fuel             0\n",
      "seller_type      0\n",
      "transmission     0\n",
      "owner            0\n",
      "dtype: int64\n",
      "Duplicated values======= 763\n",
      "After dropping the duplicated values=== 0\n",
      "Types============= name             object\n",
      "year              int64\n",
      "selling_price     int64\n",
      "km_driven         int64\n",
      "fuel             object\n",
      "seller_type      object\n",
      "transmission     object\n",
      "owner            object\n",
      "dtype: object\n",
      "==============================All the preprocessing steps are done===============================================\n"
     ]
    }
   ],
   "source": [
    "#loading the data sheet and preprocessing\n",
    "print(\"=================================loading the data sheet and preprocessing===============================\")\n",
    "df = pd.read_csv('C:\\Capstone project\\CAR DETAILS.csv')\n",
    "print(df.shape) # rows=4340,cols=8\n",
    "print(df.head())\n",
    "print(df.columns)\n",
    "# to check for null values and we found no values in this\n",
    "print(df.isnull().sum()) \n",
    "\n",
    "# To check for duplicated values.\n",
    "print(\"Duplicated values=======\",df.duplicated().sum())\n",
    "df.drop_duplicates(inplace=True)\n",
    "print(\"After dropping the duplicated values===\",df.duplicated().sum())\n",
    "print(\"Types=============\",df.dtypes)\n",
    "print(\"==============================All the preprocessing steps are done===============================================\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9b91f038",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=================================Preparing For ML Model==========================================================\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "<class 'pandas.core.series.Series'>\n",
      "(3577, 7)\n",
      "(3577,)\n",
      "(3575, 7)\n",
      "(2, 7)\n",
      "(3575,)\n",
      "(2,)\n"
     ]
    }
   ],
   "source": [
    "print(\"=================================Preparing For ML Model==========================================================\")\n",
    "# Selecting the dependent and independent features.\n",
    "x = df.drop('selling_price',axis=1)\n",
    "y = df['selling_price']\n",
    "print(type(x))\n",
    "print(type(y))\n",
    "print(x.shape)\n",
    "print(y.shape)\n",
    "\n",
    "#Split the data into train and test.\n",
    "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=2,random_state=42)\n",
    "print(x_train.shape)\n",
    "print(x_test.shape)\n",
    "print(y_train.shape)\n",
    "print(y_test.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "86f4a1ed",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission',\n",
      "       'owner'],\n",
      "      dtype='object')\n",
      "                       name  year  km_driven    fuel seller_type transmission  \\\n",
      "0             Maruti 800 AC  2007      70000  Petrol  Individual       Manual   \n",
      "1  Maruti Wagon R LXI Minor  2007      50000  Petrol  Individual       Manual   \n",
      "\n",
      "         owner  \n",
      "0  First Owner  \n",
      "1  First Owner  \n"
     ]
    }
   ],
   "source": [
    "#Evaluating the model  -R2_Score,MSE,RMSE,MAE\n",
    "\n",
    "def eval_model(ytest,ypred):\n",
    "    mae = mean_absolute_error(y_test,ypred)\n",
    "    mse = mean_squared_error(ytest,ypred)\n",
    "    rmse = np.sqrt(mean_squared_error(y_test,ypred))\n",
    "    r2s = r2_score(y_test,ypred)\n",
    "    print('MAE',mae)\n",
    "    print('MSE',mse)\n",
    "    print('RMSE',rmse)\n",
    "    print('R2S',r2s)\n",
    "\n",
    "print(x.columns)\n",
    "print(x.head(2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d2395e39",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ColumnTransformer(remainder='passthrough',\n",
      "                  transformers=[('col_transf',\n",
      "                                 OneHotEncoder(drop='first', sparse=False),\n",
      "                                 [0, 3, 4, 5, 6])])\n",
      "MAE 78586.39440207928\n",
      "MSE 6915588975.508049\n",
      "RMSE 83160.02029525996\n",
      "R2S 0.5745889134635571\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'MAE 78586.39440207928\\nMSE 6915588975.508049\\nRMSE 83160.02029525996\\nR2S 0.5745889134635571'"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Linear regression\n",
    "step1 = ColumnTransformer(transformers=\n",
    "                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],\n",
    "                          remainder='passthrough')\n",
    "print(step1)\n",
    "step2 = LinearRegression()\n",
    "pipe_lr = Pipeline([('step1',step1),('step2',step2)])\n",
    "pipe_lr.fit(x_train,y_train)\n",
    "ypred_lr = pipe_lr.predict(x_test)\n",
    "eval_model(y_test,ypred_lr)\n",
    "\"\"\"MAE 78586.39440207928\n",
    "MSE 6915588975.508049\n",
    "RMSE 83160.02029525996\n",
    "R2S 0.5745889134635571\"\"\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "bfac6b5c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ColumnTransformer(remainder='passthrough',\n",
      "                  transformers=[('col_transf',\n",
      "                                 OneHotEncoder(drop='first', sparse=False),\n",
      "                                 [0, 3, 4, 5, 6])])\n",
      "MAE 101622.54128474742\n",
      "MSE 20525153338.36775\n",
      "RMSE 143266.02297253787\n",
      "R2S -0.26260074361354846\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'MAE 101622.54128474742\\nMSE 20525153338.36775\\nRMSE 143266.02297253787\\nR2S -0.26260074361354846'"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Ridge Regression.\n",
    "step1 = ColumnTransformer(transformers=\n",
    "                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],\n",
    "                          remainder='passthrough')\n",
    "print(step1)\n",
    "step2 = Ridge(alpha=10)\n",
    "pipe_ridge = Pipeline([('step1',step1),('step2',step2)])\n",
    "pipe_ridge.fit(x_train,y_train)\n",
    "ypred_ridge = pipe_ridge.predict(x_test)\n",
    "eval_model(y_test,ypred_ridge)\n",
    "\"\"\"MAE 101622.54128474742\n",
    "MSE 20525153338.36775\n",
    "RMSE 143266.02297253787\n",
    "R2S -0.26260074361354846\"\"\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "50b8be60",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ColumnTransformer(remainder='passthrough',\n",
      "                  transformers=[('col_transf',\n",
      "                                 OneHotEncoder(drop='first', sparse=False),\n",
      "                                 [0, 3, 4, 5, 6])])\n",
      "MAE 77961.4412997365\n",
      "MSE 6861961267.121216\n",
      "RMSE 82836.95592621216\n",
      "R2S 0.5778878113266457\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\samar\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.087e+13, tolerance: 9.275e+10\n",
      "  model = cd_fast.enet_coordinate_descent(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'MAE 84129.68849280849\\nMSE 8174808695.130636\\nRMSE 90414.64867559148\\nR2S 0.4971282617374464'"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Lasso Regression\n",
    "step1 = ColumnTransformer(transformers=\n",
    "                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],\n",
    "                          remainder='passthrough')\n",
    "print(step1)\n",
    "step2 = Lasso(alpha=0.1)\n",
    "pipe_Lasso = Pipeline([('step1',step1),('step2',step2)])\n",
    "pipe_Lasso.fit(x_train,y_train)\n",
    "ypred_Lasso = pipe_Lasso.predict(x_test)\n",
    "eval_model(y_test,ypred_Lasso)\n",
    "\"\"\"MAE 84129.68849280849\n",
    "MSE 8174808695.130636\n",
    "RMSE 90414.64867559148\n",
    "R2S 0.4971282617374464\"\"\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "ad5e8893",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ColumnTransformer(remainder='passthrough',\n",
      "                  transformers=[('col_transf',\n",
      "                                 OneHotEncoder(drop='first', sparse=False),\n",
      "                                 [0, 3, 4, 5, 6])])\n",
      "MAE 160349.95\n",
      "MSE 25759028280.005005\n",
      "RMSE 160496.19397357997\n",
      "R2S -0.5845615243371014\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'MAE 160349.95\\nMSE 25759028280.005005\\nRMSE 160496.19397357997\\nR2S -0.5845615243371014'"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#KNN Regression.\n",
    "step1 = ColumnTransformer(transformers=\n",
    "                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],\n",
    "                          remainder='passthrough')\n",
    "print(step1)\n",
    "step2 = KNeighborsRegressor(n_neighbors=10)\n",
    "pipe_knn = Pipeline([('step1',step1),('step2',step2)])\n",
    "pipe_knn.fit(x_train,y_train)\n",
    "ypred_knn = pipe_knn.predict(x_test)\n",
    "eval_model(y_test,ypred_knn)\n",
    "\"\"\"MAE 160349.95\n",
    "MSE 25759028280.005005\n",
    "RMSE 160496.19397357997\n",
    "R2S -0.5845615243371014\"\"\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ae24449f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ColumnTransformer(remainder='passthrough',\n",
      "                  transformers=[('col_transf',\n",
      "                                 OneHotEncoder(drop='first', sparse=False),\n",
      "                                 [0, 3, 4, 5, 6])])\n",
      "MAE 101805.79063360882\n",
      "MSE 10600077874.32859\n",
      "RMSE 102956.6796003474\n",
      "R2S 0.34793830838424666\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'MAE 101805.79063360882\\nMSE 10600077874.32859\\nRMSE 102956.6796003474\\nR2S 0.34793830838424666'"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#DT Regression.\n",
    "step1 = ColumnTransformer(transformers=\n",
    "                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],\n",
    "                          remainder='passthrough')\n",
    "print(step1)\n",
    "step2 = DecisionTreeRegressor(max_depth=30,min_samples_split=15)\n",
    "pipe_dt = Pipeline([('step1',step1),('step2',step2)])\n",
    "pipe_dt.fit(x_train,y_train)\n",
    "ypred_dt = pipe_dt.predict(x_test)\n",
    "eval_model(y_test,ypred_dt)\n",
    "\"\"\"MAE 101805.79063360882\n",
    "MSE 10600077874.32859\n",
    "RMSE 102956.6796003474\n",
    "R2S 0.34793830838424666\"\"\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "af649424",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ColumnTransformer(remainder='passthrough',\n",
      "                  transformers=[('col_transf',\n",
      "                                 OneHotEncoder(drop='first', sparse=False),\n",
      "                                 [0, 3, 4, 5, 6])])\n",
      "MAE 101671.25045406833\n",
      "MSE 12982289800.579502\n",
      "RMSE 113939.85167876734\n",
      "R2S 0.20139701342071503\n"
     ]
    }
   ],
   "source": [
    "#Random Forest.\n",
    "step1 = ColumnTransformer(transformers=\n",
    "                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],\n",
    "                          remainder='passthrough')\n",
    "print(step1)\n",
    "step2 = RandomForestRegressor(n_estimators=100,max_depth=50,min_samples_split=15,random_state=3)\n",
    "pipe_rf = Pipeline([('step1',step1),('step2',step2)])\n",
    "pipe_rf.fit(x_train,y_train)\n",
    "ypred_rf = pipe_rf.predict(x_test)\n",
    "eval_model(y_test,ypred_rf)\n",
    "\"\"\"MAE 101671.25045406833\n",
    "MSE 12982289800.579502\n",
    "RMSE 113939.85167876734\n",
    "R2S 0.20139701342071503\"\"\"\n",
    "## Linear regression is the best model.\n",
    "\n",
    "#Saving the model.\n",
    "pickle.dump(pipe_lr,open('lrmodel.pkl','wb')) # Saving the model.\n",
    "pickle.dump(df,open('data.pkl','wb')) #Saving the data frame\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20f2b46b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "956e8092",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
