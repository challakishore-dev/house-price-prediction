
import streamlit as st, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction Dashboard")

@st.cache_data
def load():
    return pd.read_csv("data/housing.csv")
df=load()

num_cols=["area","bedrooms","bathrooms","age","parking"]
cat_cols=["location"]
X=df.drop("price",axis=1); y=df["price"]
pre=ColumnTransformer([
 ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num_cols),
 ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat_cols)
])
model=Pipeline([("pre",pre),("rf",RandomForestRegressor(n_estimators=250,random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model.fit(Xtr,ytr)
pred=model.predict(Xte)
r2=r2_score(yte,pred)
rmse=np.sqrt(mean_squared_error(yte,pred))
st.subheader("Predict Price")
col1,col2=st.columns(2)
with col1:
    area=st.slider("Area (sqft)",500,4500,1800)
    bedrooms=st.slider("Bedrooms",1,6,3)
    bathrooms=st.slider("Bathrooms",1,5,2)
with col2:
    age=st.slider("Age",0,35,5)
    parking=st.slider("Parking",0,2,1)
    location=st.selectbox("Location",["Premium","Standard","Budget"])

if st.button("Estimate Price"):
    sample=pd.DataFrame([locals()])[["area","bedrooms","bathrooms","age","parking","location"]]
    price=float(model.predict(sample)[0])
    st.success(f"Estimated Price: ₹{price:,.0f}")

st.subheader("Visualizations")
tab1,tab2=st.tabs(["Correlation","Actual vs Predicted"])
with tab1:
    fig,ax=plt.subplots(figsize=(8,4))
    sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="viridis",ax=ax)
    st.pyplot(fig)
with tab2:
    fig,ax=plt.subplots(figsize=(7,4))
    ax.scatter(yte,pred,alpha=.6)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    st.pyplot(fig)
