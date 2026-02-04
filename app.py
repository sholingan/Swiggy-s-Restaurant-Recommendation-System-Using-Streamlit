# FINAL UNIFIED SWIGGY RECOMMENDER APP

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Swiggy Recommender", layout="wide")

# ---------- SESSION STATE ----------
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("swiggy_cleaned_data.csv")
    enc = pd.read_csv("swiggy_encoded_data.csv")
    enc.index = df.index
    return df, enc

df, encoded = load_data()

# ---------- HEADER (ONLY ONCE) ----------
st.image("https://i.imghippo.com/files/Ps1522v.jpg", use_container_width=True)
st.title("🍽️ Smart Swiggy Recommender System")

# ---------- SIDEBAR (ONLY ONE) ----------
menu = st.sidebar.radio(
    "Choose Section",
    [
        "Browse Restaurants",
        "Find Similar Restaurants",
        "AI Recommendations",
        "Favorites",
        "Insights"
    ]
)

# ---------- IMAGE FUNCTION ----------
def get_image():
    return "https://img.freepik.com/free-photo/restaurant-interior_1127-3394.jpg"

# ---------- CARD DISPLAY ----------
def show_card(r):

    st.image(get_image(), width=200)

    st.write(f"### {r['name']}")
    st.write(f"📍 {r['city']}")
    st.write(f"🍴 {r['cuisine']}")
    st.write(f"⭐ {r['rating']} | 💰 ₹{r['cost']}")

    map_link = f"https://www.google.com/maps/search/?api=1&query={r['name']} {r['city']}"
    st.markdown(f"[📍 View on Google Maps]({map_link})")

    if st.button(f"❤️ Add to Favorites - {r['name']}", key=r.name):
        st.session_state.favorites.append(r["name"])
        st.success("Added to favorites!")

# ===================== PAGE 1 =====================
if menu == "Browse Restaurants":

    st.subheader("Browse Restaurants")

    city = st.selectbox("Select City", ["All"] + sorted(df["city"].unique()))

    temp = df.copy()

    if city != "All":
        temp = temp[temp["city"] == city]

    for _, r in temp.head(10).iterrows():
        show_card(r)

# ===================== PAGE 2 =====================

if menu == "Find Similar Restaurants":

    st.subheader("Find Similar Restaurants")

    query = st.text_input("Search Restaurant or Area Name")

    if query:

        matches = df[df["name"].str.contains(query, case=False)]

        if not matches.empty:

            idx = st.selectbox(
                "Select Restaurant",
                matches.index,
                format_func=lambda x: df.loc[x, "name"]
            )

            if st.button("Get Recommendations"):

                base = encoded.iloc[idx].values.reshape(1, -1)

                scores = cosine_similarity(base, encoded)[0]

                top = scores.argsort()[::-1][1:6]

                st.subheader("Recommended Restaurants")

                for i in top:
                    show_card(df.iloc[i])

        else:
            st.warning("No matching restaurants found")


    # ------------- SEARCH SECTION WITH FULL DETAILS -------------

    st.write("---")
    st.subheader("🔍 Search Restaurant to get more details")

    search = st.text_input("Enter restaurant name")

    if search:

        result = df[df["name"].str.contains(search, case=False)]

        if not result.empty:
            st.subheader("Search Results")
            st.dataframe(result)
        else:
            st.error("Restaurant not found")


    # ============ SMART RESTAURANT FINDER (WITH DROPDOWNS) =============

    st.write("----")
    st.subheader("🎯 Smart Restaurant Finder")

    # Budget Input
    budget = st.number_input("Enter Your Budget (₹)", min_value=0, step=50)

    # Area / City Dropdown
    city_list = ["All"] + sorted(df["city"].unique().tolist())
    area = st.selectbox("Select Your Area / City", city_list)

    # Cuisine Dropdown
    cuisine_list = ["All"] + sorted(df["cuisine"].unique().tolist())
    cuisine = st.selectbox("Select Cuisine Type", cuisine_list)

    if st.button("Find Restaurants"):

        temp = df.copy()

        # Budget filter
        if budget > 0:
            temp = temp[temp["cost"] <= budget]

        # Area filter
        if area != "All":
            temp = temp[temp["city"] == area]

        # Cuisine filter
        if cuisine != "All":
            temp = temp[temp["cuisine"].str.contains(cuisine, case=False)]

        if not temp.empty:
            st.subheader("Matching Restaurants")
            st.dataframe(temp[["name", "city", "rating", "cost", "cuisine"]])
        else:
            st.error("No restaurants found for given filters")


   

# ===================== PAGE 3 =====================
if menu == "AI Recommendations":

    st.subheader("Machine Learning Based Suggestions")

    model = RandomForestRegressor()
    model.fit(encoded, df["rating"])

    df["predicted"] = model.predict(encoded)

    best = df.sort_values("predicted", ascending=False).head(10)

    for _, r in best.iterrows():
        show_card(r)

# ===================== PAGE 4 =====================
if menu == "Favorites":

    st.subheader("Your Favorite Restaurants")

    fav = df[df["name"].isin(st.session_state.favorites)]

    if fav.empty:
        st.info("No favorites added yet")
    else:
        for _, r in fav.iterrows():
            show_card(r)

# ===================== PAGE 5 : INSIGHTS =====================

if menu == "Insights":

    st.subheader("Top Restaurants Filter")

    option = st.selectbox("Select Top Range", [10,20,50,100])

    top = df.sort_values("rating", ascending=False).head(option)

    st.dataframe(top[["name", "city", "rating", "cost"]])

    st.subheader("City Wise Restaurant Count")
    st.bar_chart(df["city"].value_counts().head(10))

    st.subheader("Cost Distribution")
    plt.figure(figsize=(8,4))
    plt.hist(df["cost"], bins=20)
    st.pyplot(plt.gcf())

    st.subheader("Average Rating by City")
    avg_city = df.groupby("city")["rating"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_city)

    st.subheader("Popular Cuisines")
    if "cuisine" in df.columns:
        st.bar_chart(df["cuisine"].value_counts().head(10))

    st.subheader("Cost vs Rating Analysis")
    plt.figure(figsize=(8,4))
    plt.scatter(df["cost"], df["rating"])
    plt.xlabel("Cost")
    plt.ylabel("Rating")
    st.pyplot(plt.gcf())

    st.subheader("Search Restaurant")
    search = st.text_input("Enter restaurant name")

    if search:
        result = df[df["name"].str.contains(search, case=False)]
        st.dataframe(result)

    st.subheader("Best Value Restaurants")
    best = df[(df["rating"] > 4.0)].sort_values("cost").head(10)
    st.dataframe(best[["name", "city", "rating", "cost"]])
