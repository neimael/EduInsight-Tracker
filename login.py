import streamlit as st
import app as main_page

# Set page configuration
st.set_page_config(
    page_title="EduInsight Tracker",
    page_icon="📊"
)

# Check if the user is logged in
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_page.main()
else:
    # Create two empty columns, the second one will be used to center the image
    col1, col2, col3 = st.columns([1, 2, 1])

    # Put an empty Markdown element in the first and third columns
    # col1.markdown("")
    col3.markdown("")

    # Put the image in the middle column
    with col2:
        st.image("Login.png")

    # Page title and welcome message
    st.title("Welcome to EduInsight Tracker! 👨‍🏫")
    st.subheader("Please login to access your account.")

    # Login form with placeholders
    username = st.text_input("👨‍🏫 Username", placeholder="Enter your username")
    password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

    # Login button with icon and adjusted size
    if st.button("Login", help="Click to login", key="login_button"):
        if username == "naima" and password == "teacher":
            st.success("Login successful! Redirecting...")
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password. Please try again.")
