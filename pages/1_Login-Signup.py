import streamlit as st
import bcrypt

# ------------------------------
# Session Setup
# ------------------------------
if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None


# ------------------------------
# Helper Functions
# ------------------------------
def signup(name, username, email, password):
    if not name or not username or not email or not password:
        return False, "❌ All fields are required!"
    if username in st.session_state.users:
        return False, "❌ Username already exists!"
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    st.session_state.users[username] = {
        "name": name,
        "email": email,
        "password": hashed_pw,
    }
    return True, "🎉 Account created successfully! Please login."


def login(username, password):
    user = st.session_state.users.get(username)
    if not user:
        return False, "❌ Username not found!"
    if bcrypt.checkpw(password.encode(), user["password"].encode()):
        st.session_state.logged_in = True
        st.session_state.username = username
        return True, f"✅ Welcome back, {user['name'] or username}!"
    else:
        return False, "❌ Incorrect password."


def logout():
    st.session_state.logged_in = False
    st.session_state.username = None


# ------------------------------
# Login / Signup UI
# ------------------------------
st.title("Login / Signup")

tab1, tab2 = st.tabs(["Login", "Signup"])

with tab1:
    login_username = st.text_input("Username", key="login_username")
    login_password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        success, msg = login(login_username, login_password)
        if success:
            st.success(msg)
        else:
            st.error(msg)

with tab2:
    name = st.text_input("Full Name", key="signup_name")
    signup_username = st.text_input("Choose Username", key="signup_username")
    email = st.text_input("Email", key="signup_email")
    signup_password = st.text_input("Password", type="password", key="signup_password")
    if st.button("Signup"):
        success, msg = signup(name, signup_username, email, signup_password)
        if success:
            st.success(msg)
        else:
            st.error(msg)

# If logged in, show logout
if st.session_state.logged_in:
    st.success(f"✅ You are logged in as **{st.session_state.username}**")
    if st.button("Logout"):
        logout()
        st.rerun()
