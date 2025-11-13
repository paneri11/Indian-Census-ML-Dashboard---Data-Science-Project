import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix, \
    silhouette_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# --- Page Config ---
st.set_page_config(
    page_title="Indian Census ML Dashboard",
    page_icon="🇮🇳",
    layout="wide",
)


# --- Data Loading and Caching ---
@st.cache_data(show_spinner="Loading and cleaning census data...")
def load_data():
    """
    Loads, cleans, merges, and engineers features from the 2001 and 2011 census data.
    """

    # --- Load 2001 Data (PC01_PCA_TOT_00_00.csv) ---
    try:
        # Read the first row to check columns
        df_2001_cols = pd.read_csv("PC01_PCA_TOT_00_00.csv", nrows=0, na_values=[" -   ", " -"])
        df_2001_cols.columns = df_2001_cols.columns.str.strip()

        # Load the full data
        df_2001 = pd.read_csv("PC01_PCA_TOT_00_00.csv", na_values=[" -   ", " -"])
        df_2001.columns = df_2001.columns.str.strip()

        # ADDED M/F Lit and Work columns
        cols_to_keep_2001 = {
            'LEVEL': 'LEVEL', 'NAME': 'State', 'TRU': 'TRU',
            'TOT_P': 'Tot_P_2001', 'TOT_M': 'Tot_M_2001', 'TOT_F': 'Tot_F_2001',
            'P_06': 'P_06_2001', 'P_LIT': 'Lit_P_2001', 'P_ILL': 'Ill_P_2001',
            'M_LIT': 'Lit_M_2001', 'F_LIT': 'Lit_F_2001',
            'TOT_WORK_P': 'Tot_Work_P_2001', 'TOT_WORK_M': 'Tot_Work_M_2001', 'TOT_WORK_F': 'Tot_Work_F_2001',
            'NON_WORK_P': 'Non_Work_P_2001'
        }

        missing_cols = [col for col in cols_to_keep_2001.keys() if col not in df_2001.columns]
        if missing_cols:
            st.error(f"Error: The 2001 CSV is missing expected columns: {missing_cols}")
            st.error(f"Available columns: {list(df_2001.columns)}")
            return None

        df_2001 = df_2001[cols_to_keep_2001.keys()].rename(columns=cols_to_keep_2001)
        df_2001 = df_2001[(df_2001['LEVEL'] == 'STATE') & (df_2001['TRU'] == 'Total')].copy()

        # ADDED new cols to numeric list
        num_cols_2001 = ['Tot_P_2001', 'Tot_M_2001', 'Tot_F_2001', 'P_06_2001',
                         'Lit_P_2001', 'Ill_P_2001', 'Lit_M_2001', 'Lit_F_2001',
                         'Tot_Work_P_2001', 'Tot_Work_M_2001', 'Tot_Work_F_2001',
                         'Non_Work_P_2001']

        for col in num_cols_2001:
            df_2001[col] = df_2001[col].astype(str).str.replace(r'[",]', '', regex=True).str.strip()
            df_2001[col] = pd.to_numeric(df_2001[col], errors='coerce')

    except FileNotFoundError:
        st.error("Error: `PC01_PCA_TOT_00_00.csv` (2001 data) not found.")
        return None
    except Exception as e:
        st.error(f"Error loading 2001 data: {e}")
        return None

    # --- Load 2011 Data (DDW_PCA0000_2011_Indiastatedist (1).csv) ---
    try:
        # Make loading robust like 2001: strip headers and handle na values
        df_2011 = pd.read_csv("DDW_PCA0000_2011_Indiastatedist (1).csv", na_values=[" -   ", " -"])
        df_2011.columns = df_2011.columns.str.strip()

        # ADDED M/F Lit and Work columns
        cols_to_keep_2011 = {
            'Level': 'LEVEL', 'Name': 'State', 'TRU': 'TRU',
            'TOT_P': 'Tot_P_2011', 'TOT_M': 'Tot_M_2011', 'TOT_F': 'Tot_F_2011',
            'P_06': 'P_06_2011', 'P_LIT': 'Lit_P_2011', 'P_ILL': 'Ill_P_2011',
            'M_LIT': 'Lit_M_2011', 'F_LIT': 'Lit_F_2011',
            'TOT_WORK_P': 'Tot_Work_P_2011', 'TOT_WORK_M': 'Tot_Work_M_2011', 'TOT_WORK_F': 'Tot_Work_F_2011',
            'NON_WORK_P': 'Non_Work_P_2011'
        }
        # Verify expected columns exist to give a clear error if the CSV uses unexpected names
        missing_cols_2011 = [col for col in cols_to_keep_2011.keys() if col not in df_2011.columns]
        if missing_cols_2011:
            st.error(f"Error: The 2011 CSV is missing expected columns: {missing_cols_2011}")
            st.error(f"Available columns: {list(df_2011.columns)}")
            return None

        df_2011 = df_2011[cols_to_keep_2011.keys()].rename(columns=cols_to_keep_2011)
        df_2011 = df_2011[(df_2011['LEVEL'] == 'STATE') & (df_2011['TRU'] == 'Total')].copy()

        # ADDED new cols to numeric list
        num_cols_2011 = ['Tot_P_2011', 'Tot_M_2011', 'Tot_F_2011', 'P_06_2011',
                         'Lit_P_2011', 'Ill_P_2011', 'Lit_M_2011', 'Lit_F_2011',
                         'Tot_Work_P_2011', 'Tot_Work_M_2011', 'Tot_Work_F_2011',
                         'Non_Work_P_2011']
        # Clean numeric columns (remove commas/quotes/spaces) then convert to numeric
        for col in num_cols_2011:
            df_2011[col] = df_2011[col].astype(str).str.replace(r'[",]', '', regex=True).str.strip()
            df_2011[col] = pd.to_numeric(df_2011[col], errors='coerce')

    except FileNotFoundError:
        st.error("Error: `DDW_PCA0000_2011_Indiastatedist (1).csv` (2011 data) not found.")
        return None
    except Exception as e:
        st.error(f"Error loading 2011 data: {e}")
        return None

    # --- Merge Data ---
    df_2001['State'] = df_2001['State'].str.strip().str.upper()
    df_2011['State'] = df_2011['State'].str.strip().str.upper()
    df_merged = pd.merge(df_2001, df_2011, on='State', how='inner')  # Use inner to avoid states with missing data
    df_merged = df_merged[df_merged['State'] != 'INDIA'].copy()

    # --- Feature Engineering ---
    # Population Growth
    df_merged['Pop_Growth_Rate'] = ((df_merged['Tot_P_2011'] - df_merged['Tot_P_2001']) / df_merged['Tot_P_2001']) * 100

    # 2001 Features (for predicting 2001-2011 growth)
    df_merged['Literacy_Rate_2001'] = (df_merged['Lit_P_2001'] / df_merged['Tot_P_2001']) * 100
    df_merged['Workforce_Participation_2001'] = (df_merged['Tot_Work_P_2001'] / df_merged['Tot_P_2001']) * 100
    df_merged['Child_Pop_Ratio_2001'] = (df_merged['P_06_2001'] / df_merged['Tot_P_2001']) * 100
    df_merged['Sex_Ratio_2001'] = (df_merged['Tot_F_2001'] / df_merged['Tot_M_2001']) * 1000
    df_merged['Male_Lit_Rate_2001'] = (df_merged['Lit_M_2001'] / df_merged['Tot_M_2001']) * 100
    df_merged['Female_Lit_Rate_2001'] = (df_merged['Lit_F_2001'] / df_merged['Tot_F_2001']) * 100
    df_merged['Male_Work_Rate_2001'] = (df_merged['Tot_Work_M_2001'] / df_merged['Tot_M_2001']) * 100
    df_merged['Female_Work_Rate_2001'] = (df_merged['Tot_Work_F_2001'] / df_merged['Tot_F_2001']) * 100

    # 2011 Features (for clustering, classification)
    df_merged['Literacy_Rate_2011'] = (df_merged['Lit_P_2011'] / df_merged['Tot_P_2011']) * 100
    df_merged['Workforce_Participation_2011'] = (df_merged['Tot_Work_P_2011'] / df_merged['Tot_P_2011']) * 100
    df_merged['Child_Pop_Ratio_2011'] = (df_merged['P_06_2011'] / df_merged['Tot_P_2011']) * 100
    df_merged['Sex_Ratio_2011'] = (df_merged['Tot_F_2011'] / df_merged['Tot_M_2011']) * 1000
    df_merged['Male_Lit_Rate_2011'] = (df_merged['Lit_M_2011'] / df_merged['Tot_M_2011']) * 100
    df_merged['Female_Lit_Rate_2011'] = (df_merged['Lit_F_2011'] / df_merged['Tot_F_2011']) * 100
    df_merged['Male_Work_Rate_2011'] = (df_merged['Tot_Work_M_2011'] / df_merged['Tot_M_2011']) * 100
    df_merged['Female_Work_Rate_2011'] = (df_merged['Tot_Work_F_2011'] / df_merged['Tot_F_2011']) * 100

    # Gender Gap Features (2011)
    df_merged['Lit_Gender_Gap_2011'] = df_merged['Male_Lit_Rate_2011'] - df_merged['Female_Lit_Rate_2011']
    df_merged['Work_Gender_Gap_2011'] = df_merged['Male_Work_Rate_2011'] - df_merged['Female_Work_Rate_2011']

    # Other Features
    df_merged['Non_Work_Rate_2011'] = (df_merged['Non_Work_P_2011'] / df_merged['Tot_P_2011']) * 100

    # Classification Target
    median_growth = df_merged['Pop_Growth_Rate'].median()
    df_merged['High_Growth_Class'] = (df_merged['Pop_Growth_Rate'] > median_growth).astype(int)

    # Handle potential division by zero or inf values
    df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_merged.dropna(inplace=True)

    return df_merged


# Load data
df_merged = load_data()

if df_merged is None or df_merged.empty:
    st.error("Data loading failed. Please check your CSV files and refresh.")
    st.stop()


# --- Caching for Models ---
@st.cache_resource
def get_model_assets(df):
    """
    Prepares and caches all models and scalers.
    """
    assets = {}

    # --- 1. Population Forecast Model (Simple Linear Regression) ---
    X_pop = df[['Tot_P_2001']]
    y_pop = df['Tot_P_2011']
    pop_model = LinearRegression()
    pop_model.fit(X_pop, y_pop)
    assets['pop_model'] = pop_model
    assets['pop_model_r2'] = r2_score(y_pop, pop_model.predict(X_pop))

    # --- 2. Growth Drivers & Regression Models Assets ---

    # --- Multi-Feature Models (MLR, Ridge, RF) ---
    multi_features = ['Literacy_Rate_2001', 'Workforce_Participation_2001', 'Child_Pop_Ratio_2001', 'Sex_Ratio_2001']
    X_multi = df[multi_features]
    y_multi = df['Pop_Growth_Rate']

    # *** FIX: Split data BEFORE scaling ***
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_multi, y_multi, test_size=0.3, random_state=42)

    # *** FIX: Fit scaler ONLY on training data ***
    growth_scaler = StandardScaler()
    X_m_train_scaled = growth_scaler.fit_transform(X_m_train)
    # *** FIX: Transform test data with the same scaler ***
    X_m_test_scaled = growth_scaler.transform(X_m_test)

    # Model 2a: Multiple Linear Regression (for Growth Drivers page)
    growth_model = LinearRegression()
    growth_model.fit(X_m_train_scaled, y_m_train)  # Train on scaled data

    # Store assets for Growth Drivers page
    assets['growth_model'] = growth_model
    assets['growth_scaler'] = growth_scaler  # The scaler is fit and ready
    assets['growth_features'] = multi_features
    # Calculate metrics on the SCALED test set
    assets['growth_model_r2'] = r2_score(y_m_test, growth_model.predict(X_m_test_scaled))
    assets['growth_model_mae'] = mean_absolute_error(y_m_test, growth_model.predict(X_m_test_scaled))

    # Model 2b: Ridge Regression
    assets['ridge_model'] = Ridge(alpha=1.0).fit(X_m_train_scaled, y_m_train)

    # Model 2c: Random Forest Regressor
    # Note: RF doesn't strictly need scaling, but it's fine to use it for consistency.
    assets['rf_regressor_model'] = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_m_train_scaled,
                                                                                                y_m_train)

    # Store multi-feature test data (scaled)
    assets['reg_multi_test_data'] = (X_m_test_scaled, y_m_test)  # Use scaled test data
    assets['reg_multi_features'] = multi_features

    # --- Simple Linear Model ---
    simple_feature = 'Child_Pop_Ratio_2001'  # Using the most correlated feature
    X_simple = df[[simple_feature]]
    y_simple = df['Pop_Growth_Rate']

    # Simple model doesn't need scaling, so we split normally
    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_simple, y_simple, test_size=0.3, random_state=42)

    simple_model = LinearRegression().fit(X_s_train, y_s_train)
    assets['simple_model'] = simple_model
    assets['simple_model_feature'] = simple_feature
    assets['simple_model_test_data'] = (X_s_test, y_s_test)

    # --- 3. Classification Model Assets ---
    class_features = ['Literacy_Rate_2011', 'Workforce_Participation_2011', 'Child_Pop_Ratio_2011', 'Sex_Ratio_2011']
    X_class = df[class_features]
    y_class = df['High_Growth_Class']

    # *** FIX: Split data BEFORE scaling ***
    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

    # *** FIX: Fit scaler ONLY on training data ***
    class_scaler = StandardScaler()
    X_c_train_scaled = class_scaler.fit_transform(X_c_train)
    # *** FIX: Transform test data with the same scaler ***
    X_c_test_scaled = class_scaler.transform(X_c_test)

    assets['class_data'] = (X_c_train_scaled, X_c_test_scaled, y_c_train, y_c_test, class_features)
    assets['class_scaler'] = class_scaler  # Store the fit scaler

    # Train models on scaled data
    assets['log_reg_model'] = LogisticRegression().fit(X_c_train_scaled, y_c_train)
    assets['dec_tree_model'] = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_c_train_scaled, y_c_train)

    # --- 4. Clustering Model Assets ---
    cluster_features = ['Literacy_Rate_2011', 'Workforce_Participation_2011', 'Child_Pop_Ratio_2011', 'Sex_Ratio_2011',
                        'Lit_Gender_Gap_2011', 'Work_Gender_Gap_2011']
    X_cluster = df[cluster_features]

    # For clustering, we scale the *entire* dataset because we're not testing, we're exploring.
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

    assets['cluster_data'] = X_cluster_scaled
    assets['cluster_features'] = cluster_features
    assets['cluster_scaler'] = cluster_scaler

    return assets


# Get all models and assets
model_assets = get_model_assets(df_merged)

# --- Sidebar Navigation ---
st.sidebar.title("🇮🇳 Census ML Dashboard")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📊 Exploratory Data Analysis (EDA)",
    "🤖 Model Performance",
    "💡 Growth Drivers Analysis",
    "🔮 Population Prediction"
])

# ==============================================================================
# --- Page 1: Home ---
# ==============================================================================
if page == "🏠 Home":
    st.title("Welcome to the Indian Census ML Dashboard")
    st.markdown("""
    This application analyzes and models population data from the 2001 and 2011 Indian Censuses.

    - **Datasets:** `PC01_PCA_TOT_00_00.csv` (2001) and `DDW_PCA0000_2011_Indiastatedist (1).csv` (2011)
    - **Goal:** To find insights, build predictive models, and forecast population trends.

    Use the sidebar to navigate between the different analysis pages.
    """)

    with st.container(border=True):
        st.header("Key National Statistics (2001 vs 2011)", divider="rainbow")

        pop_2001 = df_merged['Tot_P_2001'].sum()
        pop_2011 = df_merged['Tot_P_2011'].sum()
        growth = pop_2011 - pop_2001
        growth_rate = (growth / pop_2001) * 100

        lit_2001 = df_merged['Lit_P_2001'].sum()
        lit_rate_2001 = (lit_2001 / pop_2001) * 100

        lit_2011 = df_merged['Lit_P_2011'].sum()
        lit_rate_2011 = (lit_2011 / pop_2011) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Population (2001)", f"{pop_2001:,.0f}")
        col1.metric("Total Population (2011)", f"{pop_2011:,.0f}")

        col2.metric("Decadal Growth (Absolute)", f"{growth:,.0f}")
        col2.metric("Decadal Growth (Rate)", f"{growth_rate:.2f}%")

        col3.metric("Literacy Rate (2001)", f"{lit_rate_2001:.2f}%")
        col3.metric("Literacy Rate (2011)", f"{lit_rate_2011:.2f}%")

    with st.container(border=True):
        st.header("Merged & Processed Data (Sample)", divider="rainbow")
        st.dataframe(df_merged.head())
        st.markdown(f"Successfully loaded and merged data for **{df_merged.shape[0]}** states and union territories.")


# ==============================================================================
# --- Page 2: Exploratory Data Analysis (EDA) ---
# ==============================================================================
elif page == "📊 Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Visualizing the relationships and distributions in the census data. (13 visualizations total)")

    with st.container(border=True):
        st.subheader("1. Population Growth: 2001 vs 2011", divider="rainbow")
        st.markdown("This chart shows the total population for each state in 2001 and 2011.")

        df_plot = df_merged[['State', 'Tot_P_2001', 'Tot_P_2011']].melt(
            id_vars='State', var_name='Year', value_name='Population'
        )
        df_plot['Year'] = df_plot['Year'].str.replace('Tot_P_', '')

        fig1 = px.bar(
            df_plot.sort_values('Population'),
            x='Population', y='State',
            color='Year', barmode='group',
            title='Population by State: 2001 vs 2011',
            height=800
        )
        st.plotly_chart(fig1, use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Literacy Rate vs. Population Growth (2011)", divider="rainbow")
        st.markdown("This bubble chart explores the relationship between literacy, growth rate, and population size.")

        fig2 = px.scatter(
            df_merged,
            x='Literacy_Rate_2011',
            y='Pop_Growth_Rate',
            size='Tot_P_2011',
            color='State',
            hover_name='State',
            size_max=60,
            title='Literacy Rate (2011) vs. Decadal Population Growth Rate (%)'
        )
        fig2.update_layout(xaxis_title="Literacy Rate 2011 (%)", yaxis_title="Population Growth Rate (2001-2011) (%)")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
        **Insight:** You can often see a trend where states with higher literacy rates tend to have lower population growth rates. The size of the bubble represents the state's total 2011 population.
        """)

    with st.container(border=True):
        st.subheader("3. Demographic Correlation Heatmap", divider="rainbow")
        st.markdown("Shows which variables are correlated. A value close to 1 or -1 indicates a strong relationship.")

        corr_cols = [
            'Pop_Growth_Rate', 'Literacy_Rate_2011', 'Workforce_Participation_2011',
            'Child_Pop_Ratio_2011', 'Sex_Ratio_2011', 'Lit_Gender_Gap_2011', 'Work_Gender_Gap_2011'
        ]
        corr_matrix = df_merged[corr_cols].corr()
        fig3 = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap of Key Demographic Factors"
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
        **Insight:** Look for strong correlations. For example, `Child_Pop_Ratio_2011` (percentage of children) is often strongly positively correlated with `Pop_Growth_Rate` and negatively correlated with `Literacy_Rate_2011`.
        """)

    # --- 10 NEW EDAs START HERE ---

    with st.container(border=True):
        st.subheader("4. Top & Bottom 10 States by Population Growth Rate (2001-2011)", divider="rainbow")
        st.markdown("Which states grew the fastest and slowest in the last decade?")

        df_sorted = df_merged.sort_values('Pop_Growth_Rate', ascending=False)
        df_top_bottom = pd.concat([df_sorted.head(10), df_sorted.tail(10)])

        fig4 = px.bar(
            df_top_bottom,
            x='Pop_Growth_Rate',
            y='State',
            color='Pop_Growth_Rate',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            color_continuous_midpoint=df_merged['Pop_Growth_Rate'].mean(),
            title='Population Growth Rate (%) by State',
            orientation='h'
        )
        fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""
        **Insight:** This chart clearly identifies the states with the highest (e.g., Meghalaya, Arunachal Pradesh) and lowest (e.g., Nagaland, Kerala) decadal population growth rates.
        """)

    with st.container(border=True):
        st.subheader("5. Change in Sex Ratio (2001 vs 2011)", divider="rainbow")
        st.markdown("A dumbbell plot showing the change in sex ratio (females per 1000 males) for each state.")

        fig5 = go.Figure()
        for i, row in df_merged.iterrows():
            fig5.add_trace(go.Scatter(
                x=[row['Sex_Ratio_2001'], row['Sex_Ratio_2011']],
                y=[row['State'], row['State']],
                mode='lines+markers',
                line=dict(color='grey', width=1),
                marker=dict(size=8),
                name=row['State'],
                hoverinfo='none'
            ))

        fig5.add_trace(go.Scatter(
            x=df_merged['Sex_Ratio_2001'], y=df_merged['State'],
            mode='markers', name='2001', marker=dict(color='#d62728', size=10)
        ))
        fig5.add_trace(go.Scatter(
            x=df_merged['Sex_Ratio_2011'], y=df_merged['State'],
            mode='markers', name='2011', marker=dict(color='#2ca02c', size=10)
        ))

        fig5.update_layout(
            title='Change in Sex Ratio (Females per 1000 Males) from 2001 to 2011',
            xaxis_title='Sex Ratio',
            yaxis_title='State',
            height=800,
            showlegend=True,
            yaxis=dict(categoryorder='total ascending')
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("""
        **Insight:** This chart helps visualize whether the sex ratio improved (green dot > red dot) or worsened (red dot > green dot) for each state over the decade.
        """)

    with st.container(border=True):
        st.subheader("6. Change in Overall Literacy Rate (2001 vs 2011)", divider="rainbow")
        st.markdown("A dumbbell plot showing the improvement in total literacy rate.")

        fig6 = go.Figure()
        # Add lines
        for i, row in df_merged.iterrows():
            fig6.add_trace(go.Scatter(
                x=[row['Literacy_Rate_2001'], row['Literacy_Rate_2011']],
                y=[row['State'], row['State']],
                mode='lines',
                line=dict(color='grey', width=1),
                hoverinfo='none'
            ))

        # Add points
        fig6.add_trace(go.Scatter(
            x=df_merged['Literacy_Rate_2001'], y=df_merged['State'],
            mode='markers', name='2001', marker=dict(color='#d62728', size=10)
        ))
        fig6.add_trace(go.Scatter(
            x=df_merged['Literacy_Rate_2011'], y=df_merged['State'],
            mode='markers', name='2011', marker=dict(color='#2ca02c', size=10)
        ))

        fig6.update_layout(
            title='Change in Literacy Rate (%) from 2001 to 2011',
            xaxis_title='Literacy Rate (%)',
            yaxis_title='State',
            height=800,
            showlegend=True,
            yaxis=dict(categoryorder='total ascending')
        )
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("""
        **Insight:** We can see that literacy rates improved in nearly every state. The length of the line shows the magnitude of the improvement.
        """)

    with st.container(border=True):
        st.subheader("7. Literacy Gender Gap (2011)", divider="rainbow")
        st.markdown("Which states have the largest gap between male and female literacy rates?")

        fig7 = px.bar(
            df_merged.sort_values('Lit_Gender_Gap_2011', ascending=True),
            y='State',
            x='Lit_Gender_Gap_2011',
            title='Literacy Gender Gap (Male % - Female %) in 2011',
            labels={'Lit_Gender_Gap_2011': 'Literacy Gap (Percentage Points)'},
            height=800,
            orientation='h'
        )
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("""
        **Insight:** States with longer bars have a greater disparity in literacy between men and women. States with the smallest gaps (like Meghalaya and Kerala) are closer to gender parity in education.
        """)

    with st.container(border=True):
        st.subheader("8. Workforce Gender Gap (2011)", divider="rainbow")
        st.markdown("Which states have the largest gap between male and female workforce participation?")

        fig8 = px.bar(
            df_merged.sort_values('Work_Gender_Gap_2011', ascending=True),
            y='State',
            x='Work_Gender_Gap_2011',
            title='Workforce Participation Gender Gap (Male % - Female %) in 2011',
            labels={'Work_Gender_Gap_2011': 'Workforce Gap (Percentage Points)'},
            height=800,
            orientation='h'
        )
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("""
        **Insight:** This often shows an even larger gap than literacy. It highlights the economic disparity between genders across the country.
        """)

    with st.container(border=True):
        st.subheader("9. Male vs. Female Workforce Participation (2011)", divider="rainbow")
        st.markdown("A scatter plot showing the relationship between male and female work rates.")

        fig9 = px.scatter(
            df_merged,
            x='Male_Work_Rate_2011',
            y='Female_Work_Rate_2011',
            color='State',
            hover_name='State',
            title='Male vs. Female Workforce Participation Rate (2011)',
            labels={
                'Male_Work_Rate_2011': 'Male Workforce Participation (%)',
                'Female_Work_Rate_2011': 'Female Workforce Participation (%)'
            }
        )
        # Add a y=x line for reference (parity)
        fig9.add_shape(
            type="line",
            x0=0, y0=0, x1=100, y1=100,
            line=dict(color="Grey", width=2, dash="dash"),
            name='Parity Line'
        )
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown("""
        **Insight:** Each dot is a state. The dashed line represents perfect equality (Male Rate = Female Rate). Almost all states fall far below this line, showing significantly lower female workforce participation.
        """)

    with st.container(border=True):
        st.subheader("10. Distribution of Key Metrics (2011)", divider="rainbow")
        st.markdown("View the spread, median, and outliers for key demographics.")

        metric_to_plot = st.selectbox(
            "Select a metric to see its distribution:",
            ('Pop_Growth_Rate', 'Literacy_Rate_2011', 'Sex_Ratio_2011', 'Child_Pop_Ratio_2011', 'Lit_Gender_Gap_2011',
             'Work_Gender_Gap_2011'),
            index=0
        )

        fig10 = px.histogram(
            df_merged,
            x=metric_to_plot,
            marginal="box",  # Adds a box plot on top
            title=f'Distribution of {metric_to_plot} Across States (2011)'
        )
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown("""
        **Insight:** The histogram shows the frequency of different values. The box plot at the top shows the median (center line), the interquartile range (the box), and outliers (dots). This helps you understand the "normal" range for a metric and which states are exceptions.
        """)

    with st.container(border=True):
        st.subheader("11. Child Population vs. Population Growth (2011)", divider="rainbow")
        st.markdown("Examines the link between the 0-6 age group ratio and population growth.")

        fig11 = px.scatter(
            df_merged,
            x='Child_Pop_Ratio_2011',
            y='Pop_Growth_Rate',
            color='State',
            hover_name='State',
            trendline='ols',  # Add an "ordinary least squares" trendline
            title='Child Population Ratio (2011) vs. Decadal Growth Rate',
            labels={
                'Child_Pop_Ratio_2011': 'Child (0-6) Population (% of Total)',
                'Pop_Growth_Rate': 'Population Growth Rate (2001-2011) (%)'
            }
        )
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown("""
        **Insight:** The strong positive trendline suggests that a higher proportion of young children is a powerful predictor of a higher overall population growth rate.
        """)

    with st.container(border=True):
        st.subheader("12. Workforce Participation vs. Literacy (2011)", divider="rainbow")
        st.markdown("Is there a link between a state's literacy and its workforce?")

        fig12 = px.scatter(
            df_merged,
            x='Literacy_Rate_2011',
            y='Workforce_Participation_2011',
            color='State',
            hover_name='State',
            trendline='ols',
            title='Literacy Rate vs. Workforce Participation (2011)',
            labels={
                'Literacy_Rate_2011': 'Literacy Rate (%)',
                'Workforce_Participation_2011': 'Workforce Participation Rate (%)'
            }
        )
        st.plotly_chart(fig12, use_container_width=True)
        st.markdown("""
        **Insight:** The relationship here is often weak or ambiguous. This tells us that literacy alone doesn't directly translate to a higher workforce participation rate. Other factors (like gender gaps, as seen before) are at play.
        """)

    with st.container(border=True):
        st.subheader("13. National Worker vs. Non-Worker Share (2001 vs 2011)", divider="rainbow")
        st.markdown("How has the overall economic dependency ratio changed in a decade?")

        # Calculate national totals
        total_work_2001 = df_merged['Tot_Work_P_2001'].sum()
        total_non_work_2001 = df_merged['Non_Work_P_2001'].sum()

        total_work_2011 = df_merged['Tot_Work_P_2011'].sum()
        total_non_work_2011 = df_merged['Non_Work_P_2011'].sum()

        df_pie_2001 = pd.DataFrame({
            'Category': ['Workers', 'Non-Workers'],
            'Population': [total_work_2001, total_non_work_2001]
        })
        df_pie_2011 = pd.DataFrame({
            'Category': ['Workers', 'Non-Workers'],
            'Population': [total_work_2011, total_non_work_2011]
        })

        col1, col2 = st.columns(2)
        with col1:
            fig13a = px.pie(
                df_pie_2001,
                names='Category',
                values='Population',
                title='National Worker Share (2001)',
                hole=0.3
            )
            st.plotly_chart(fig13a, use_container_width=True)
        with col2:
            fig13b = px.pie(
                df_pie_2011,
                names='Category',
                values='Population',
                title='National Worker Share (2011)',
                hole=0.3
            )
            st.plotly_chart(fig13b, use_container_width=True)

        st.markdown("""
        **Insight:** These pie charts show the "big picture" of the workforce. We can see if the proportion of workers to non-workers (which includes children, students, and the elderly) has changed over the decade, indicating shifts in the national economic dependency ratio.
        """)


# ==============================================================================
# --- Page 3: Model Performance ---
# ==============================================================================
elif page == "🤖 Model Performance":
    st.header("🤖 Model Performance Evaluation")
    st.markdown("Evaluating our Regression, Classification, and Clustering models.")

    # Create new tabs, adding Regression first
    reg_tab, class_tab, cluster_tab = st.tabs(["📈 Regression Models", "🤖 Classification Models", "📊 Clustering Models"])

    # --- NEW REGRESSION TAB ---
    with reg_tab:
        st.subheader("Goal: Predict the 'Population Growth Rate (%)'")
        st.markdown("""
        Here we compare different regression models to see how accurately we can predict
        the `Pop_Growth_Rate` (from 2001-2011) using the 2001 demographic data.
        """)

        reg_model_choice = st.selectbox(
            "Select Regression Model:",
            ["Simple Linear Regression", "Multiple Linear Regression", "Ridge Regression", "Random Forest Regressor"]
        )

        model = None
        y_pred = None
        y_test = None

        st.divider()

        # --- Simple Linear Regression ---
        if reg_model_choice == "Simple Linear Regression":
            model = model_assets['simple_model']
            X_test, y_test = model_assets['simple_model_test_data']
            feature_name = model_assets['simple_model_feature']

            st.markdown(f"**Model:** Predicts `Pop_Growth_Rate` using *only* one feature: `{feature_name}`.")
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            col1, col2 = st.columns(2)
            col1.metric("R-squared (R²)", f"{r2:.3f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}%")

            st.subheader("Predicted vs. Actual")
            plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Feature': X_test[feature_name]})
            fig = px.scatter(
                plot_df, x='Feature', y='Actual',
                title=f"Actual Growth vs. {feature_name}",
                labels={'Feature': feature_name, 'Actual': 'Actual Growth Rate (%)'}
            )
            # Add the regression line
            fig.add_trace(go.Scatter(x=plot_df['Feature'], y=plot_df['Predicted'], mode='lines', name='Model Fit',
                                     line=dict(color='red')))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("The red line shows the model's learned relationship. The dots show the actual data.")

        # --- Common block for Multi-Feature Models ---
        else:
            # *** Retrieve the correctly scaled test data ***
            X_test_scaled, y_test = model_assets['reg_multi_test_data']
            features = model_assets['reg_multi_features']

            if reg_model_choice == "Multiple Linear Regression":
                model = model_assets['growth_model']
                st.markdown(f"**Model:** Standard linear model using 4 features: `{', '.join(features)}`.")

            elif reg_model_choice == "Ridge Regression":
                model = model_assets['ridge_model']
                st.markdown(
                    f"**Model:** A 'regularized' linear model that is more robust to co-linear features. Also uses all 4 features.")

            elif reg_model_choice == "Random Forest Regressor":
                model = model_assets['rf_regressor_model']
                st.markdown(
                    f"**Model:** A powerful, non-linear tree-based model. Uses all 4 features to find complex patterns.")

            # *** Predict using the scaled test data ***
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            col1, col2 = st.columns(2)
            col1.metric("R-squared (R²)", f"{r2:.3f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}%")

            # *** Add explanation for R-squared ***
            if r2 < 0:
                st.warning(f"""
                **Note on R-squared:** A negative R² value ({r2:.3f}) means the model is performing
                worse on this small test set than a simple horizontal line (just guessing the average).
                This is common with very small datasets (~10 test samples) where a few outliers
                can heavily penalize the score. It highlights the model's "overfitting" on the training data.
                """)

            st.subheader("Predicted vs. Actual")
            plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

            fig = px.scatter(
                plot_df, x='Actual', y='Predicted',
                title='Predicted vs. Actual Population Growth Rate',
                labels={'Actual': 'Actual Growth Rate (%)', 'Predicted': 'Predicted Growth Rate (%)'},
                marginal_x="histogram", marginal_y="histogram"
            )
            # Add a y=x line for reference (perfect prediction)
            fig.add_shape(
                type="line",
                x0=plot_df['Actual'].min(), y0=plot_df['Actual'].min(),
                x1=plot_df['Actual'].max(), y1=plot_df['Actual'].max(),
                line=dict(color="Red", width=2, dash="dash"),
                name='Perfect Prediction'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "The red dashed line represents a perfect prediction. The closer the dots are to this line, the better the model.")

            # Add feature importance for Random Forest
            if reg_model_choice == "Random Forest Regressor":
                st.subheader("Feature Importance")
                st.markdown("Which features did the Random Forest find most important for predicting growth?")
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                 title='Random Forest Feature Importance')
                st.plotly_chart(fig_imp, use_container_width=True)

    # --- Classification Tab ---
    with class_tab:
        st.subheader("Goal: Predict if a state has 'High' or 'Low' Population Growth")
        st.markdown("""
        We use a state's 2011 demographics (Literacy, Workforce, Sex Ratio, etc.) to predict if its
        decadal growth rate was 'High' (1) or 'Low' (0), relative to the median.
        """)

        model_choice = st.selectbox("Select Classification Model:", ["Logistic Regression", "Decision Tree"])

        # *** Retrieve the correctly scaled train/test data ***
        X_c_train_scaled, X_c_test_scaled, y_c_train, y_c_test, features = model_assets['class_data']

        model = None
        if model_choice == "Logistic Regression":
            model = model_assets['log_reg_model']
        else:
            model = model_assets['dec_tree_model']

        # *** Predict on scaled test data ***
        y_pred = model.predict(X_c_test_scaled)
        acc = accuracy_score(y_c_test, y_pred)

        st.metric(f"Model Accuracy", f"{acc * 100:.2f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Classification Report")
            st.code(classification_report(y_c_test, y_pred), language='text')

        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_c_test, y_pred)
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted Label", y="True Label"),
                x=['Low Growth (0)', 'High Growth (1)'],
                y=['Low Growth (0)', 'High Growth (1)'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        if model_choice == "Decision Tree":
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                             title='Decision Tree Feature Importance')
            st.plotly_chart(fig_imp, use_container_width=True)

    # --- Clustering Tab ---
    with cluster_tab:
        st.subheader("Goal: Group states with similar demographic profiles")
        st.markdown("""
        We use K-Means Clustering (an unsupervised model) to group states based on their 2011
        demographics. This helps identify "clusters" of similar states.
        """)

        X_cluster_scaled = model_assets['cluster_data']
        cluster_features = model_assets['cluster_features']

        k = st.slider("Select Number of Clusters (k)", min_value=2, max_value=8, value=3)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_cluster_scaled)

        # Do not mutate df_merged in-place; create a copy for visualization
        df_clustered = df_merged.copy()
        df_clustered['Cluster'] = clusters.astype(str)

        score = silhouette_score(X_cluster_scaled, clusters)
        st.metric("Silhouette Score", f"{score:.3f}")
        st.markdown("_A score closer to 1 indicates well-defined, dense clusters._")

        st.subheader("Cluster Visualization")
        st.markdown("How the clusters are separated based on two selected features.")

        col1, col2 = st.columns(2)
        x_axis = col1.selectbox("Select X-Axis Feature:", cluster_features, index=0)
        y_axis = col2.selectbox("Select Y-Axis Feature:", cluster_features, index=1)

        fig_cluster = px.scatter(
            df_clustered,
            x=x_axis,
            y=y_axis,
            color='Cluster',
            hover_name='State',
            title=f'K-Means Clusters (k={k})'
        )
        st.plotly_chart(fig_cluster, use_container_width=True)


# ==============================================================================
# --- Page 4: Growth Drivers Analysis ---
# ==============================================================================
elif page == "💡 Growth Drivers Analysis":
    st.header("💡 Growth Drivers Analysis")
    st.markdown("""
    This is our most meaningful predictive model. It uses **Multiple Linear Regression** to
    predict the **Decadal Population Growth Rate (%)** based on a state's demographics at the *start* of the decade (2001).

    **Formula:** `Growth Rate ≈ m1(Literacy) + m2(Workforce) + m3(Child Pop) + m4(Sex Ratio) + c`

    This helps us understand *why* growth happens. For a comparison with other models, see the "Model Performance" page.
    """)

    # Get model assets
    model = model_assets['growth_model']
    scaler = model_assets['growth_scaler']  # The scaler is already fit
    features = model_assets['growth_features']
    r2 = model_assets['growth_model_r2']
    mae = model_assets['growth_model_mae']

    with st.container(border=True):
        st.subheader("Model Performance (on Test Set)")
        col1, col2 = st.columns(2)
        col1.metric("R-squared (R²)", f"{r2:.3f}")
        col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}%")
        st.markdown(
            f"This model can predict the decadal growth rate, on average, within **{mae:.2f} percentage points** (on the test data).")
        if r2 < 0:
            st.warning(
                f"**Note on R-squared:** The negative R² value ({r2:.3f}) indicates the model performed poorly on the small test set. This is a risk with very small datasets. The coefficients below are based on the *training* data and are still useful for interpretation.")

    with st.container(border=True):
        st.subheader("What-If Simulator")
        st.markdown("See how changing a state's 2001 demographics could have impacted its 2001-2011 growth rate.")

        # Select a state to pre-fill sliders
        state_list = sorted(df_merged['State'].unique())
        default_state = "BIHAR"
        default_index = state_list.index(default_state) if default_state in state_list else 0

        selected_state = st.selectbox("Select a State to use as a baseline:", state_list, index=default_index)
        baseline = df_merged[df_merged['State'] == selected_state].iloc[0]

        # Create sliders, pre-filled with baseline data
        lit_2001 = st.slider("Literacy Rate 2001 (%)", 0.0, 100.0, baseline['Literacy_Rate_2001'], 0.1)
        work_2001 = st.slider("Workforce Participation 2001 (%)", 0.0, 100.0, baseline['Workforce_Participation_2001'],
                              0.1)
        child_2001 = st.slider("Child (0-6) Population Ratio 2001 (%)", 0.0, 50.0, baseline['Child_Pop_Ratio_2001'],
                               0.1)
        sex_2001 = st.slider("Sex Ratio 2001 (Females per 1000 Males)", 700.0, 1200.0, baseline['Sex_Ratio_2001'], 1.0)

        # Prepare data for prediction
        # We must use the 'scaler' that the model was trained on
        input_data = np.array([[lit_2001, work_2001, child_2001, sex_2001]])
        # *** Use the fit scaler to transform the new data ***
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        predicted_growth = model.predict(input_data_scaled)[0]

        st.metric("Predicted Decadal Growth Rate (2001-2011)", f"{predicted_growth:.2f}%")
        st.markdown(f"**Actual Growth Rate for {selected_state}:** `{baseline['Pop_Growth_Rate']:.2f}%`")

    with st.container(border=True):
        st.subheader("Growth Drivers (Model Coefficients)")
        st.markdown("""
        These "coefficients" are the core of the model. They show the impact each factor has on the growth rate, *after* scaling.
        - **Negative Value:** An *increase* in this factor *decreases* the population growth. (e.g., higher literacy)
        - **Positive Value:** An *increase* in this factor *increases* the population growth. (e.g., higher child population)
        """)

        coeffs = pd.DataFrame({
            'Feature': features,
            'Coefficient (Impact)': model.coef_
        }).sort_values(by='Coefficient (Impact)', ascending=False)

        fig_coeffs = px.bar(
            coeffs, x='Coefficient (Impact)', y='Feature', orientation='h',
            title="Impact of Demographics on Population Growth",
            color='Coefficient (Impact)',
            color_continuous_scale=px.colors.diverging.RdBu,
            color_continuous_midpoint=0
        )
        st.plotly_chart(fig_coeffs, use_container_width=True)


# ==============================================================================
# --- Page 5: Population Prediction ---
# ==============================================================================
elif page == "🔮 Population Prediction":
    st.header("🔮 Population Forecast (2021 & 2031)")
    st.markdown("""
    This page forecasts the population for 2021 and 2031 using a *simple* time-series model.
    It learns the growth pattern from 2001 to 2011 and applies it forward.

    **Model:** `Population(t) = m * Population(t-1) + c`

    - **2021 Prediction:** `Pop(2021) = m * Pop(2011) + c`
    - **2031 Prediction:** `Pop(2031) = m * Pop(2021) + c`

    *For a more detailed explanation of *why* growth happens, see the **"💡 Growth Drivers Analysis"** page.*
    """)

    pop_model = model_assets['pop_model']

    with st.container(border=True):
        st.subheader("Population Model Performance (Trained on 2001 -> 2011)")
        st.metric("R-squared (R²)", f"{model_assets['pop_model_r2']:.4f}")
        st.markdown(f"""
        - **Coefficient (m):** `{pop_model.coef_[0]:.4f}` (For every 1 person in 2001, there were ~{pop_model.coef_[0]:.2f} in 2011)
        - **Intercept (c):** `{pop_model.intercept_:,.0f}` (A baseline population offset)
        """)

    # --- Perform Predictions ---
    df_pred = df_merged[['State', 'Tot_P_2001', 'Tot_P_2011']].copy()

    # Predict 2021
    # We rename the 2011 column to 'Tot_P_2001' so it matches what the model was trained on
    X_2021_input = df_pred[['Tot_P_2011']].rename(columns={'Tot_P_2011': 'Tot_P_2001'})
    df_pred['Pred_P_2021'] = pop_model.predict(X_2021_input)

    # Predict 2031
    # We rename the 2021 column to 'Tot_P_2001' for the same reason
    X_2031_input = df_pred[['Pred_P_2021']].rename(columns={'Pred_P_2021': 'Tot_P_2001'})
    df_pred['Pred_P_2031'] = pop_model.predict(X_2031_input)

    # Clean and format
    df_pred['Pred_P_2021'] = df_pred['Pred_P_2021'].astype(int)
    df_pred['Pred_P_2031'] = df_pred['Pred_P_2031'].astype(int)

    with st.container(border=True):
        st.subheader("Population Forecast Results")
        st.dataframe(
            df_pred.sort_values('Tot_P_2011', ascending=False),
            column_config={
                "Tot_P_2001": st.column_config.NumberColumn(format="%,d"),
                "Tot_P_2011": st.column_config.NumberColumn(format="%,d"),
                "Pred_P_2021": st.column_config.NumberColumn(format="%,d"),
                "Pred_P_2031": st.column_config.NumberColumn(format="%,d"),
            }
        )

    with st.container(border=True):
        st.subheader("Forecast by State")

        state_list = sorted(df_pred['State'].unique())
        default_state = "UTTAR PRADESH"
        default_index = state_list.index(default_state) if default_state in state_list else 0

        state_to_plot = st.selectbox("Select State to Visualize Forecast:", state_list, index=default_index)

        plot_data = df_pred[df_pred['State'] == state_to_plot].iloc[0]

        forecast_df = pd.DataFrame({
            'Year': [2001, 2011, 2021, 2031],
            'Population': [
                plot_data['Tot_P_2001'],
                plot_data['Tot_P_2011'],
                plot_data['Pred_P_2021'],
                plot_data['Pred_P_2031']
            ],
            'Type': ['Actual', 'Actual', 'Predicted', 'Predicted']
        })

        fig_forecast = px.line(
            forecast_df, x='Year', y='Population',
            color='Type', title=f'Population Forecast for {state_to_plot}',
            markers=True
        )
        fig_forecast.update_layout(xaxis=dict(tickmode='linear', dtick=10))
        st.plotly_chart(fig_forecast, use_container_width=True)

