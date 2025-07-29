import streamlit as st
import joblib as jl
import sqlite3

conn = sqlite3.connect("house_predictor.db")
cursor = conn.cursor()


cursor.execute("""
    CREATE TABLE IF NOT EXISTS Primary_Features(
               Area  INTEGER not null,
               Bedrooms INTEGER not null,
               Bathrooms INTEGER not null,
               Stories INTEGER not null
               )
""")


cursor.execute("""
    CREATE TABLE IF NOT EXISTS Secondary_Features(
               Mainroad INTEGER not null,
               Guestroom INTEGER not null,
               Basement INTEGER not null,
               Airconditioning INTEGER not null,
               Parking INTEGER not null,
               Prefarea INTEGER not null,
               Furnishingstatus INTEGER not null,
               Predictions INTEGER not null
               )

""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS User_info(
               id INTEGER Primary Key,
               Name TEXT not null,
               Email TEXT not null
               )
""")





def insert_Primary(*args):
    cursor.execute("INSERT into Primary_Features (Area,Bedrooms,Bathrooms,Stories) values (?,?,?,?)",args)
    conn.commit()

def insert_Secondary(*args):
    cursor.execute("INSERT into Secondary_Features (Mainroad,Guestroom,Basement,Airconditioning,Parking,Prefarea,Furnishingstatus,Predictions) values (?,?,?,?,?,?,?,?)",args)
    conn.commit()

def insert_User(*args):
    cursor.execute("INSERT into User_info (Name,Email) values (?,?)",args)
    conn.commit()





def Show_Primary():
    primary_value = cursor.execute("SELECT * FROM Primary_Features").fetchall()
    for pri in primary_value:
        st.success(f"Area = {pri[0]} , Bedrooms = {pri[1]},  Bathrooms = {pri[2]},  Stories = {pri[3]}")

def Show_Secondary():
    secondary_value = cursor.execute("Select * FROM Secondary_Features").fetchall()
    for sec in secondary_value:
        st.success(f"Mainroad = {'Yes' if  sec[0]==1 else 'No'} , Guestroom = {'Yes' if  sec[1]==1 else 'No'}" 
                f", Basement = {'Yes' if  sec[2]==1 else 'No'},  Airconditioning = {'Yes' if  sec[3]==1 else 'No'}" 
                f", Parking = {'Yes' if  sec[4]==1 else 'No'},  PrefArea = {'Yes' if  sec[5]==1 else 'No'}"
                f", Furnishingstatus = {'Furnished' if  sec[6]==1 else 'Not-Furnished'} , Predicted Price = {sec[7]}"
                )


def show_User():
    user_data = cursor.execute("Select * from User_info").fetchall()
    for user in user_data:
        st.success(f"Id: {user[0]}  - Name:      {user[1]} -   Email:       {user[2]}")



st.title("House Price Predictor")

area = st.slider("Enter the Area of House:",7000,15000,8500)
bedrooms = st.number_input("Number of Bedrooms:",min_value=2,max_value=8,step=1)
bathrooms = st.number_input("Number of Bathrooms:",min_value=2,max_value=8,step=1)
stories = st.number_input("Number of Stories:",min_value=2,max_value=3,step=1)

st.success("Select 1 for Required, 0 for Not-Required")

main_road =st.number_input("On Main Raod:",min_value=0,max_value=1,step=1)
guest_room =st.number_input("Guest Room required:",min_value=0,max_value=1,step=1)
basement = st.number_input("Basement Required:",min_value=0,max_value=1,step=1)
air_conditioning = st.number_input("Air Conditioning Required:",min_value=0,max_value=1,step=1)
parking = st.number_input("Parking required:",min_value=0,max_value=2,step=1)
pref_area = st.number_input("Pref Area Required:",min_value=0,max_value=1,step=1)
furnished = st.number_input("Furnished or Not:",min_value=0,max_value=1,step=1)

st.subheader("Model:")
model_selection = st.selectbox("Select Model",['Linear Regression','Random Forest'])

with st.sidebar.form("form"):
    st.markdown('### Enter your Deatils')
    name = st.text_input("Enter your Name: ")
    email = st.text_input("Enter your Email: ")
    submit = st.form_submit_button("Submit")

    if submit:
        if name!="" and email!="":
            insert_User(name,email)
            st.success("Thanks, Your Details are saved")
        elif name!=" " or email!=" ":
            st.success("Kindly Fill the Credentials..!!")


users = st.button("Show Users")
if users:
    show_User()

   




st.subheader("Features")
feature = st.selectbox("" ,["Select a Category","Primary","Secondary"])

if feature=="Primary":
    Show_Primary()

elif feature=="Secondary":
    Show_Secondary()
else:
    st.success("Select Feature from Menu .. !! ")

b3 = st.button("Predict")
if b3:

    if model_selection == 'Linear Regression':
        st.success("Model is predicting results......")
       
        model = jl.load("Linear_Regression_Model.pkl")
        sc = jl.load('scaler.pkl')
        import numpy as np
        custom_Data = np.array([[area,bedrooms,bathrooms,stories,main_road,guest_room,basement,air_conditioning,parking,pref_area,furnished]])
    
        # Scaler
        std_data = sc.transform(custom_Data)

        model_prediction = model.predict(std_data)
        predicted_price = int(np.exp(model_prediction)[0])  # Convert to integer
        st.subheader(f"💰 The Prediction of Linear Regression is approximately: PKR {predicted_price:,}")
        insert_Primary(area,bedrooms,bathrooms,stories)
        insert_Secondary(main_road,guest_room,basement,air_conditioning,parking,pref_area,furnished,predicted_price)


    else:
        st.success("Model is predicting results......")
        model = jl.load("Trained_Rf_model.pkl")
        import numpy as np
        custom_Data = np.array([[area,bedrooms,bathrooms,stories,main_road,guest_room,basement,air_conditioning,parking,pref_area,furnished]])
        model_prediction = model.predict(custom_Data)
        predicted_price = int(np.exp(model_prediction)[0])  # Convert to integer
        st.subheader(f"💰 The Prediction of Random Forest is approximately: PKR {predicted_price:,}")
        insert_Primary(area,bedrooms,bathrooms,stories)
        insert_Secondary(main_road,guest_room,basement,air_conditioning,parking,pref_area,furnished,predicted_price)





