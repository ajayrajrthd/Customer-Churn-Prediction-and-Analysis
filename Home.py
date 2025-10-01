import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Churn Analysis & Prediction"
)

# --- Custom CSS for Dark Mode Effects and Animations ---

st.markdown("""
<style>
.hero {
    text-align: center;
    animation: fadeIn 2s ease-in-out;
}
.hero h1 {
    font-size: 3.5em;
    color: #FFA500;
    font-weight: 700;
    margin-bottom: 15px;
    transition: transform 0.5s ease, color 0.5s ease;
}
.hero h1:hover {
    transform: scale(1.05);
    color: #FF4500;
}
.hero p {
    font-size: 1.5em;
    color: #EEE;
    margin-bottom: 50px;
}
.feature-card {
    background: linear-gradient(145deg, #1f1f1f, #2c2c2c);
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.7);
    padding: 25px;
    margin-bottom: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
    color: #fff;
    text-align: center;
}
.feature-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 15px 30px rgba(255,165,0,0.7);
    background: linear-gradient(145deg, #2c2c2c, #3d3d3d);
}
.feature-card h3 {
    color: #FFA500;
    margin-bottom: 15px;
    font-size: 1.5em;
}
.feature-card p {
    color: #DDD;
    font-size: 1em;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("<h1> 📊 Welcome to Customer Churn Prediction & Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p>Predict customer churn and explore business insights effortlessly with AI-powered tools.</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Features Section ---
features = [
    {"title": "Churn Prediction", "desc": "Predict which customers are likely to churn with advanced ML and DL models."},
    {"title": "Sales Trend Analysis", "desc": "Visualize monthly revenue and churn trends to make informed decisions."},
    {"title": "Interactive Dashboard", "desc": "Explore KPIs, customer statistics, and feature analysis with interactive charts."},
    {"title": "Model Comparison", "desc": "Compare SVM, RandomForest, DecisionTree, XGBoost, ANN, and RNN models easily."},
    {"title": "Secure Access", "desc": "Login/signup authentication ensures authorized access to sensitive pages."},
    {"title": "User-Friendly UI", "desc": "Modern animations, smooth transitions, and responsive design for better experience."}
]

# Arrange cards in rows of 3
for i in range(0, len(features), 3):
    cols = st.columns(3)
    for idx, feature in enumerate(features[i:i+3]):
        with cols[idx]:
            st.markdown(f"""
                <div class="feature-card">
                    <h3>{feature['title']}</h3>
                    <p>{feature['desc']}</p>
                </div>
            """, unsafe_allow_html=True)

# --- Call to Action ---
st.markdown('<div class="hero">', unsafe_allow_html=True)

st.markdown("<p style='font-size:1.3em;color:#FFA500;'>Use the sidebar to navigate and start exploring insights now!</p>", unsafe_allow_html=True)

# --- Get Started Button ---
if st.button("🔐 Get Started"):
    st.switch_page("pages/1_Login-Signup.py")