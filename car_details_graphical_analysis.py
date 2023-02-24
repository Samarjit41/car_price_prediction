{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c372eb69",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd # data preprocessing\n",
    "import numpy as np #mathematical computation\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d84db181",
   "metadata": {},
   "outputs": [],
   "source": [
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
    "print(\"Types=============\",df.dtypes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1c6129e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# To check for outliers.\n",
    "print(df.describe())\n",
    "\n",
    "# EDA analysis.\n",
    "sns.countplot(y=df['fuel']) # Count for fuel type\n",
    "print(plt.show())\n",
    "sns.countplot(y=df['seller_type']) # Count for fuel type\n",
    "print(plt.show())\n",
    "sns.countplot(y=df['transmission']) # Count for fuel type\n",
    "print(plt.show())\n",
    "sns.countplot(y=df['owner']) # Count for fuel type\n",
    "print(plt.show())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76cf1a3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Checking for Correlations amongs the columns.\n",
    "corr = df.corr()\n",
    "#corr = corr[corr>0.7] \n",
    "sns.heatmap(corr,annot=True,cmap='RdBu')\n",
    "#plt.show()\n",
    "print(\"==============================All the preprocessing steps are done===============================================\")"
   ]
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
