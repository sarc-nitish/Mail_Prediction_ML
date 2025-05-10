import streamlit as st
import pickle

# Loading the vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('Mail_Prediction_Model.pkl', 'rb') as f:
    model = pickle.load(f)


st.write("Hi👋 I'm Nitish, This is my 2nd ML Project")
# Create the heding name
st.title("Mail Spam Classifier")
# create tha instruct for write the mail
st.write("Enter your email text and check whether it's spam or not.")

# Enter the mail
input_mail = st.text_area("Enter the mail content here:" ,height=200)

if st.button("Predict"):
    if input_mail.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize the input mail
        input_data = vectorizer.transform([input_mail])
        prediction = model.predict(input_data)

        # Output result
        if prediction[0] == 0:
            st.error("This is a SPAM mail")
        else:
            st.success("This is a HAM (Genuine) mail")

