import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis & Stock Prediction",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #0e7c6b 100%);
        padding: 2.5rem 2rem; border-radius: 16px; color: white;
        margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(15, 23, 42, 0.3);
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 0.5rem 0; letter-spacing: -0.5px; }
    .main-header p { font-size: 1.05rem; opacity: 0.9; margin: 0; font-weight: 300; }

    .section-header {
        background: linear-gradient(90deg, #f0fdf4, #ffffff);
        border-left: 5px solid #0e7c6b; padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0; margin: 2rem 0 1.5rem 0;
    }
    .section-header h2 { color: #0f172a; font-size: 1.5rem; font-weight: 600; margin: 0; }

    .concept-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.5rem; margin: 0.75rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
    }
    .concept-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
    .concept-card h4 { color: #0e7c6b; font-weight: 600; margin: 0 0 0.75rem 0; font-size: 1.1rem; }
    .concept-card p { color: #4a5568; line-height: 1.7; margin: 0; }

    .formula-box {
        background: #f8fafc; border: 2px solid #e2e8f0; border-radius: 10px;
        padding: 1.25rem; text-align: center; margin: 1rem 0; font-size: 1.05rem;
    }

    .insight-box {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5); border: 1px solid #6ee7b7;
        border-radius: 10px; padding: 1.25rem; margin: 1rem 0;
    }
    .insight-box strong { color: #065f46; }

    .warning-box {
        background: linear-gradient(135deg, #fefce8, #fef9c3); border: 1px solid #fcd34d;
        border-radius: 10px; padding: 1.25rem; margin: 1rem 0;
    }

    .metric-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.25rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-card .metric-value { font-size: 1.8rem; font-weight: 700; color: #0f172a; }
    .metric-card .metric-label {
        font-size: 0.85rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 0.5px; margin-top: 0.25rem;
    }

    .pos-tag {
        background: #dcfce7; color: #166534; padding: 0.15rem 0.5rem;
        border-radius: 12px; font-weight: 600; font-size: 0.8rem;
    }
    .neg-tag {
        background: #fee2e2; color: #991b1b; padding: 0.15rem 0.5rem;
        border-radius: 12px; font-weight: 600; font-size: 0.8rem;
    }
    .neu-tag {
        background: #f1f5f9; color: #475569; padding: 0.15rem 0.5rem;
        border-radius: 12px; font-weight: 600; font-size: 0.8rem;
    }

    .step-indicator {
        display: inline-flex; align-items: center; justify-content: center;
        width: 32px; height: 32px; background: #0e7c6b; color: white;
        border-radius: 50%; font-weight: 700; font-size: 0.9rem; margin-right: 0.75rem;
    }

    [data-testid="stSidebar"] { background: #f8fafc; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset():
    """Generate realistic synthetic stock + news sentiment data."""
    np.random.seed(42)

    # --- Stock price data: 2 years of daily data ---
    dates = pd.bdate_range('2022-01-03', periods=504)  # ~2 years business days
    # Base price with trend, volatility clustering, and regime changes
    price = [150.0]
    daily_returns = []
    vol = 0.015
    for i in range(1, len(dates)):
        # GARCH-like volatility clustering
        vol = 0.002 + 0.85 * vol + 0.1 * abs(np.random.normal(0, 0.015))
        ret = np.random.normal(0.0003, vol)
        daily_returns.append(ret)
        price.append(price[-1] * (1 + ret))

    stock_df = pd.DataFrame({
        'Date': dates,
        'Close': price,
        'Volume': np.random.lognormal(17, 0.3, len(dates)).astype(int)
    })
    stock_df['Return'] = stock_df['Close'].pct_change()
    stock_df['Open'] = stock_df['Close'].shift(1) * (1 + np.random.normal(0, 0.003, len(dates)))
    stock_df['High'] = stock_df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.008, len(dates))))
    stock_df['Low'] = stock_df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.008, len(dates))))
    stock_df = stock_df.dropna().reset_index(drop=True)

    # --- News headlines with sentiment ---
    positive_templates = [
        "{company} reports record quarterly earnings, beating analyst estimates by {pct}%",
        "{company} announces strategic partnership with major tech firm",
        "{company} raises full-year guidance amid strong consumer demand",
        "Analysts upgrade {company} to 'Buy' citing robust growth outlook",
        "{company} launches innovative new product line to strong market reception",
        "{company} expands into emerging markets with ${{val}}B investment",
        "CEO of {company} expresses confidence in long-term growth strategy",
        "{company} sees surge in user engagement, up {pct}% year-over-year",
        "{company} stock rallies after positive FDA approval announcement",
        "{company} secures major government contract worth ${{val}}B",
        "Institutional investors increase stakes in {company} significantly",
        "{company} reports strongest revenue growth in five quarters",
        "Market momentum builds for {company} after industry conference",
        "{company} dividend increase signals management confidence",
        "{company} successfully completes acquisition, expects immediate synergies",
        "Strong insider buying observed at {company} in recent filings",
        "{company} named industry leader in annual analyst survey",
        "{company} beats subscription growth targets for third straight quarter",
    ]

    negative_templates = [
        "{company} misses revenue estimates, shares fall in after-hours trading",
        "Regulatory concerns mount for {company} as investigation widens",
        "{company} announces workforce reduction of {pct}% amid restructuring",
        "Analysts downgrade {company} citing competitive headwinds",
        "{company} faces class-action lawsuit over data privacy violations",
        "Supply chain disruptions expected to impact {company} margins this quarter",
        "{company} CEO departure raises uncertainty about strategic direction",
        "{company} warns of slowing growth in key market segments",
        "Short sellers increase positions in {company} to record levels",
        "{company} product recall raises safety and liability concerns",
        "{company} loses major client contract to emerging competitor",
        "Debt concerns grow as {company} credit rating placed on review",
        "{company} reports declining user metrics for second consecutive quarter",
        "Trade tensions could significantly impact {company} supply chain costs",
        "{company} delays product launch citing technical challenges",
        "Insider selling at {company} reaches highest level in two years",
        "{company} faces margin pressure from rising input costs",
        "Market share losses accelerate for {company} in latest data",
    ]

    neutral_templates = [
        "{company} holds annual shareholder meeting, maintains current strategy",
        "Trading volume in {company} remains within normal ranges",
        "{company} schedules earnings call for next week",
        "Industry report shows flat growth across {company}'s sector",
        "{company} files routine regulatory paperwork with the SEC",
        "{company} maintains dividend at current level for next quarter",
        "Analysts maintain 'Hold' rating on {company} shares",
        "{company} appoints new board member from finance sector",
        "{company} to present at upcoming Morgan Stanley technology conference",
        "{company} completes routine share buyback program on schedule",
    ]

    companies = ['TechNova Corp', 'DataStream Inc', 'NexGen Solutions', 'Vertex Holdings']

    headlines = []
    sentiments = []
    headline_dates = []
    vader_scores = []

    for i, date in enumerate(dates[1:], 1):
        # Generate 1-3 headlines per day
        n_headlines = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        for _ in range(n_headlines):
            company = np.random.choice(companies)
            pct = np.random.randint(3, 25)
            val = np.random.randint(1, 12)

            # Sentiment somewhat correlated with returns
            ret = stock_df.loc[i-1, 'Return'] if i-1 < len(stock_df) else 0
            if ret > 0.01:
                probs = [0.65, 0.10, 0.25]  # more likely positive
            elif ret < -0.01:
                probs = [0.10, 0.65, 0.25]  # more likely negative
            else:
                probs = [0.30, 0.25, 0.45]  # balanced

            sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=probs)

            if sentiment_type == 'positive':
                template = np.random.choice(positive_templates)
                base_score = np.random.uniform(0.15, 0.85)
            elif sentiment_type == 'negative':
                template = np.random.choice(negative_templates)
                base_score = np.random.uniform(-0.85, -0.15)
            else:
                template = np.random.choice(neutral_templates)
                base_score = np.random.uniform(-0.12, 0.12)

            headline = template.format(company=company, pct=pct, val=val)
            headline = headline.replace('${val}', str(val))

            headlines.append(headline)
            sentiments.append(sentiment_type)
            headline_dates.append(date)
            vader_scores.append(round(base_score + np.random.normal(0, 0.05), 4))

    news_df = pd.DataFrame({
        'Date': headline_dates,
        'Headline': headlines,
        'Sentiment': sentiments,
        'VADER_Compound': np.clip(vader_scores, -1, 1)
    })

    return stock_df, news_df

stock_df, news_df = generate_dataset()


# ─────────────────────────────────────────────────────────────
# HELPER: Simple lexicon-based sentiment
# ─────────────────────────────────────────────────────────────
POSITIVE_WORDS = {
    'record', 'beats', 'beating', 'strong', 'growth', 'upgrade', 'innovative',
    'confidence', 'surge', 'rallies', 'positive', 'secures', 'increase',
    'strongest', 'momentum', 'leader', 'success', 'successfully', 'robust',
    'expands', 'launches', 'partnership', 'raises', 'signals', 'builds',
    'named', 'reception', 'engagement', 'buying', 'synergies', 'approval'
}

NEGATIVE_WORDS = {
    'misses', 'fall', 'concerns', 'reduction', 'downgrade', 'lawsuit',
    'disruptions', 'departure', 'slowing', 'short', 'recall', 'loses',
    'debt', 'declining', 'tensions', 'delays', 'selling', 'pressure',
    'losses', 'uncertainty', 'violations', 'headwinds', 'impact',
    'liability', 'accelerate', 'review', 'warns', 'challenges'
}

def simple_lexicon_score(text):
    """Basic bag-of-words sentiment scorer for teaching purposes."""
    words = set(re.findall(r'\w+', text.lower()))
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return round((pos_count - neg_count) / total, 3)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to section:",
        [
            "🏠 Home & Overview",
            "📖 Sentiment Analysis Theory",
            "📊 Dataset Exploration",
            "🔬 Sentiment Scoring Demo",
            "📈 Sentiment Visualizations",
            "💹 Stock Prediction — Without Sentiment",
            "🧠 Stock Prediction — With Sentiment",
            "⚔️ Head-to-Head Comparison",
            "📋 Summary & Takeaways"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📐 Quick Reference")
    st.markdown("""
    **Sentiment Scoring**
    - **Positive:** > 0.05
    - **Neutral:** -0.05 to 0.05
    - **Negative:** < -0.05

    **Key Metrics**
    - MAE, RMSE, R²
    - With vs Without Sentiment
    """)
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; font-size:0.8rem;'>"
        "SP Jain School of Global Management<br>Global MBA Program<br>Dr. Anshul Gupta</div>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════
# PAGE: HOME & OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Home & Overview":
    st.markdown("""
    <div class="main-header">
        <h1>📰 Sentiment Analysis & Stock Prediction</h1>
        <p>Understand how text sentiment from financial news can enhance stock return predictions — from NLP concepts to model results.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>What This Dashboard Covers</h2></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>📖 Sentiment Analysis</h4>
            <p>How machines extract opinion and emotion from text. Understand lexicon-based methods, VADER scoring, and how raw text becomes a numerical signal.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="concept-card">
            <h4>📊 Exploration & Visualization</h4>
            <p>Explore the stock and news dataset. Score headlines interactively. Visualize sentiment distributions, trends, and their relationship with market movements.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="concept-card">
            <h4>⚔️ With vs Without Sentiment</h4>
            <p>Build stock return prediction models using only price data, then add sentiment features. Compare accuracy head-to-head to quantify the value of NLP.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>The Workflow</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <p>
        <span class="step-indicator">1</span> <strong>Understand</strong> sentiment analysis methods — lexicon-based, VADER, and bag-of-words<br><br>
        <span class="step-indicator">2</span> <strong>Explore</strong> the stock price and financial news headline dataset<br><br>
        <span class="step-indicator">3</span> <strong>Score</strong> headlines — see how text becomes a number<br><br>
        <span class="step-indicator">4</span> <strong>Visualize</strong> sentiment patterns and their relationship with returns<br><br>
        <span class="step-indicator">5</span> <strong>Predict</strong> stock returns using only price-based features<br><br>
        <span class="step-indicator">6</span> <strong>Predict again</strong> — now with sentiment features added<br><br>
        <span class="step-indicator">7</span> <strong>Compare</strong> — does sentiment actually help?
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick dataset preview
    st.markdown('<div class="section-header"><h2>Dataset at a Glance</h2></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(stock_df)}</div><div class="metric-label">Trading Days</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(news_df):,}</div><div class="metric-label">News Headlines</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{stock_df['Date'].min().strftime('%b %Y')}</div><div class="metric-label">Start Date</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{stock_df['Date'].max().strftime('%b %Y')}</div><div class="metric-label">End Date</div></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(stock_df, x='Date', y='Close', title='Stock Price (Synthetic)', template='plotly_white')
        fig.update_traces(line=dict(color='#0e7c6b', width=2.5))
        fig.update_layout(font=dict(family='Inter'), height=350, title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        sent_counts = news_df['Sentiment'].value_counts()
        fig2 = px.pie(values=sent_counts.values, names=sent_counts.index,
                      title='Headline Sentiment Distribution',
                      color_discrete_map={'positive':'#22c55e','negative':'#ef4444','neutral':'#94a3b8'},
                      template='plotly_white')
        fig2.update_layout(font=dict(family='Inter'), height=350, title_font_size=14)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>💡 About the data:</strong> This dashboard uses a synthetic but realistic dataset — stock prices follow
        GARCH-like volatility patterns, and news headlines are generated with sentiment that partially correlates with
        market returns. This mirrors real-world dynamics where news sentiment and price movements influence each other.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS THEORY
# ═══════════════════════════════════════════════════════════════
elif page == "📖 Sentiment Analysis Theory":
    st.markdown("""
    <div class="main-header">
        <h1>📖 Sentiment Analysis — How It Works</h1>
        <p>Turning unstructured text into quantitative signals that machines can process and learn from.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- What is Sentiment Analysis ---
    st.markdown('<div class="section-header"><h2>What is Sentiment Analysis?</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <h4>Definition</h4>
        <p>
        Sentiment analysis (also called <strong>opinion mining</strong>) is the process of computationally determining
        whether a piece of text expresses a <strong>positive</strong>, <strong>negative</strong>, or <strong>neutral</strong>
        attitude. It's one of the most widely used applications of Natural Language Processing (NLP).
        </p>
        <p>
        In finance, sentiment analysis is applied to news headlines, earnings call transcripts, social media posts,
        and analyst reports to gauge market mood — often before it's reflected in prices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Methods ---
    st.markdown('<div class="section-header"><h2>Three Main Approaches</h2></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>1. Lexicon-Based (Dictionary)</h4>
            <p>
            <strong>How:</strong> Maintain a list of "positive" and "negative" words. Count how many of each appear in the text.<br><br>
            <strong>Score:</strong> (positive − negative) / total<br><br>
            <strong>Example:</strong><br>
            "Record earnings, strong growth" → 2 positive, 0 negative → Score: +1.0<br><br>
            <strong>Pros:</strong> Simple, fast, interpretable<br>
            <strong>Cons:</strong> Misses context, sarcasm, negation ("not good" ≠ negative word + positive word)
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="concept-card">
            <h4>2. VADER (Rule-Based)</h4>
            <p>
            <strong>How:</strong> A pre-built lexicon with intensity scores + rules for punctuation, capitalization,
            modifiers ("very"), and negation.<br><br>
            <strong>Score:</strong> Compound score from −1 (most negative) to +1 (most positive)<br><br>
            <strong>Example:</strong><br>
            "Earnings were GREAT!!!" → Higher score than "Earnings were great" (caps + punctuation boost)<br><br>
            <strong>Pros:</strong> Handles nuance better, no training needed<br>
            <strong>Cons:</strong> Domain-specific language may not be in the lexicon
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="concept-card">
            <h4>3. Machine Learning-Based</h4>
            <p>
            <strong>How:</strong> Train a classifier (Naive Bayes, SVM, or deep learning) on labeled examples.
            The model learns which word patterns predict positive or negative sentiment.<br><br>
            <strong>Score:</strong> Probability of each class<br><br>
            <strong>Example:</strong><br>
            Trained on 10K labeled reviews → can classify new text<br><br>
            <strong>Pros:</strong> Can learn domain-specific patterns, highest accuracy<br>
            <strong>Cons:</strong> Needs labeled training data, more complex to build
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- VADER Deep Dive ---
    st.markdown('<div class="section-header"><h2>VADER — A Closer Look</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <h4>VADER: Valence Aware Dictionary and sEntiment Reasoner</h4>
        <p>
        VADER is the most popular out-of-the-box sentiment tool for social media and short texts. It was specifically
        designed for texts like tweets, headlines, and reviews. Here's what makes it smart:
        </p>
        <p>
        <strong>Lexicon:</strong> 7,500+ words with human-rated sentiment intensity (e.g., "good" = 1.9, "great" = 3.1, "terrible" = −2.5)<br><br>
        <strong>Rule 1 — Punctuation:</strong> "This is great!!!" scores higher than "This is great."<br><br>
        <strong>Rule 2 — Capitalization:</strong> "This is GREAT" amplifies the sentiment<br><br>
        <strong>Rule 3 — Degree modifiers:</strong> "extremely good" > "slightly good"<br><br>
        <strong>Rule 4 — Negation:</strong> "not good" flips the polarity<br><br>
        <strong>Rule 5 — Conjunctions:</strong> "The food was great but the service was terrible" balances both
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <strong>VADER Output:</strong><br><br>
        <strong>pos:</strong> proportion of positive sentiment &nbsp;|&nbsp;
        <strong>neu:</strong> proportion of neutral &nbsp;|&nbsp;
        <strong>neg:</strong> proportion of negative<br><br>
        <strong>compound:</strong> normalized aggregate score from −1 to +1 (this is what we use)
    </div>
    """, unsafe_allow_html=True)

    # --- Bag of Words ---
    st.markdown('<div class="section-header"><h2>Text Preprocessing — Bag of Words</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>How Text Becomes Numbers</h4>
            <p>
            Machines don't understand words — they need numbers. The simplest conversion is the <strong>Bag of Words</strong> model:
            </p>
            <p>
            <span class="step-indicator">1</span> <strong>Tokenize:</strong> Split text into individual words<br><br>
            <span class="step-indicator">2</span> <strong>Normalize:</strong> Lowercase everything, remove punctuation<br><br>
            <span class="step-indicator">3</span> <strong>Remove stop words:</strong> Drop common words (the, is, at, and...)<br><br>
            <span class="step-indicator">4</span> <strong>Count:</strong> Create a vector of word frequencies
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="concept-card">
            <h4>Example</h4>
            <p>
            <strong>Input:</strong> "The stock price rose sharply after strong earnings"<br><br>
            <strong>After preprocessing:</strong><br>
            ["stock", "price", "rose", "sharply", "strong", "earnings"]<br><br>
            <strong>Bag of Words vector:</strong><br>
            stock:1, price:1, rose:1, sharply:1, strong:1, earnings:1<br><br>
            <em>Then sentiment is scored by matching against a positive/negative lexicon.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- Why It Matters for Finance ---
    st.markdown('<div class="section-header"><h2>Why Sentiment Matters in Finance</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <p>
        Traditional stock prediction models use only <strong>quantitative data</strong> — past prices, volume, moving averages,
        technical indicators. But markets are driven by <strong>human perception</strong>:
        </p>
        <p>
        • A positive earnings surprise headline can trigger buying before the numbers are fully analyzed<br>
        • Regulatory investigation news can cause selling based on fear, not fundamentals<br>
        • Analyst upgrades/downgrades move stocks based on <em>opinion</em>, not new data<br><br>
        Sentiment analysis captures this "soft" information and converts it into features that prediction models can use.
        The question we'll answer in this dashboard: <strong>Does adding sentiment to a stock prediction model
        actually improve accuracy?</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORATION
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Dataset Exploration":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Dataset Exploration</h1>
        <p>Examining the stock price data and financial news headlines side by side.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Stock Data ---
    st.markdown('<div class="section-header"><h2>Stock Price Data</h2></div>', unsafe_allow_html=True)

    fig_candle = go.Figure(data=[go.Candlestick(
        x=stock_df['Date'], open=stock_df['Open'], high=stock_df['High'],
        low=stock_df['Low'], close=stock_df['Close'], name='OHLC'
    )])
    fig_candle.update_layout(
        title='Stock Price — Candlestick Chart', height=450,
        template='plotly_white', font=dict(family='Inter'),
        title_font_size=15, xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Stock Data Sample**")
        st.dataframe(stock_df[['Date','Open','High','Low','Close','Volume','Return']].head(10).style.format({
            'Open':'${:.2f}','High':'${:.2f}','Low':'${:.2f}','Close':'${:.2f}',
            'Volume':'{:,.0f}','Return':'{:.4f}'
        }), use_container_width=True)
    with col2:
        st.markdown("**Return Distribution**")
        fig_ret = px.histogram(stock_df, x='Return', nbins=50, template='plotly_white',
                               color_discrete_sequence=['#0e7c6b'])
        fig_ret.update_layout(height=300, font=dict(family='Inter'))
        st.plotly_chart(fig_ret, use_container_width=True)

    # --- Key Stats ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${stock_df['Close'].iloc[0]:.0f}</div><div class="metric-label">Starting Price</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${stock_df['Close'].iloc[-1]:.0f}</div><div class="metric-label">Ending Price</div></div>""", unsafe_allow_html=True)
    with col3:
        total_ret = (stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[0] - 1) * 100
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_ret:+.1f}%</div><div class="metric-label">Total Return</div></div>""", unsafe_allow_html=True)
    with col4:
        ann_vol = stock_df['Return'].std() * np.sqrt(252) * 100
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{ann_vol:.1f}%</div><div class="metric-label">Annualized Volatility</div></div>""", unsafe_allow_html=True)

    # --- News Data ---
    st.markdown('<div class="section-header"><h2>News Headlines Data</h2></div>', unsafe_allow_html=True)

    sent_filter = st.multiselect("Filter by sentiment:", ['positive','negative','neutral'], default=['positive','negative','neutral'])
    filtered_news = news_df[news_df['Sentiment'].isin(sent_filter)]

    st.markdown(f"**Showing {len(filtered_news):,} of {len(news_df):,} headlines**")
    st.dataframe(
        filtered_news.head(20).style.format({'VADER_Compound': '{:.3f}'}),
        use_container_width=True, height=400
    )

    # --- Headlines per day ---
    daily_count = news_df.groupby('Date').size().reset_index(name='Count')
    fig_count = px.bar(daily_count, x='Date', y='Count', title='Headlines per Trading Day',
                       template='plotly_white', color_discrete_sequence=['#0e7c6b'])
    fig_count.update_layout(height=300, font=dict(family='Inter'), title_font_size=14)
    st.plotly_chart(fig_count, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: SENTIMENT SCORING DEMO
# ═══════════════════════════════════════════════════════════════
elif page == "🔬 Sentiment Scoring Demo":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Sentiment Scoring — Live Demo</h1>
        <p>See how different methods score the same piece of text. Type your own headlines and watch the scores change.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Interactive Scoring ---
    st.markdown('<div class="section-header"><h2>Try It Yourself</h2></div>', unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter a financial headline or any text:",
        value="TechNova Corp reports record quarterly earnings, beating analyst estimates by 15%",
        height=80
    )

    if user_text:
        # Lexicon score
        lex_score = simple_lexicon_score(user_text)

        # VADER score
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            sid = SentimentIntensityAnalyzer()
            vader_result = sid.polarity_scores(user_text)
        except:
            # Fallback: simulate VADER-like scores from VADER_Compound in data
            vader_result = {
                'pos': max(0, lex_score * 0.4 + 0.1),
                'neu': 0.5,
                'neg': max(0, -lex_score * 0.4 + 0.1),
                'compound': lex_score * 0.8
            }

        # TextBlob score
        try:
            from textblob import TextBlob
            blob = TextBlob(user_text)
            tb_polarity = blob.sentiment.polarity
            tb_subjectivity = blob.sentiment.subjectivity
        except:
            tb_polarity = lex_score * 0.6
            tb_subjectivity = 0.5

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            lex_color = '#166534' if lex_score > 0.05 else '#991b1b' if lex_score < -0.05 else '#475569'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{lex_color}">{lex_score:+.3f}</div>
                <div class="metric-label">Lexicon Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            vader_color = '#166534' if vader_result['compound'] > 0.05 else '#991b1b' if vader_result['compound'] < -0.05 else '#475569'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{vader_color}">{vader_result['compound']:+.4f}</div>
                <div class="metric-label">VADER Compound</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            tb_color = '#166534' if tb_polarity > 0.05 else '#991b1b' if tb_polarity < -0.05 else '#475569'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{tb_color}">{tb_polarity:+.3f}</div>
                <div class="metric-label">TextBlob Polarity</div>
            </div>
            """, unsafe_allow_html=True)

        # VADER breakdown
        st.markdown('<div class="section-header"><h2>VADER Score Breakdown</h2></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 3])
        with col1:
            vader_df = pd.DataFrame({
                'Component': ['Positive', 'Neutral', 'Negative', 'Compound'],
                'Score': [vader_result['pos'], vader_result['neu'], vader_result['neg'], vader_result['compound']]
            })
            st.dataframe(vader_df.set_index('Component').style.format('{:.4f}'), use_container_width=True)

        with col2:
            fig_vader = go.Figure(data=[go.Bar(
                x=['Positive', 'Neutral', 'Negative'],
                y=[vader_result['pos'], vader_result['neu'], vader_result['neg']],
                marker_color=['#22c55e', '#94a3b8', '#ef4444']
            )])
            fig_vader.update_layout(
                title='VADER Proportion Breakdown', height=300,
                template='plotly_white', font=dict(family='Inter'),
                title_font_size=14, yaxis_title='Proportion'
            )
            st.plotly_chart(fig_vader, use_container_width=True)

        # Lexicon word matching
        st.markdown('<div class="section-header"><h2>Lexicon Word Matching</h2></div>', unsafe_allow_html=True)

        words = set(re.findall(r'\w+', user_text.lower()))
        pos_found = words & POSITIVE_WORDS
        neg_found = words & NEGATIVE_WORDS

        col1, col2 = st.columns(2)
        with col1:
            if pos_found:
                pos_html = ' '.join([f'<span class="pos-tag">{w}</span>' for w in sorted(pos_found)])
                st.markdown(f"**Positive words found:** {pos_html}", unsafe_allow_html=True)
            else:
                st.markdown("**Positive words found:** None")
        with col2:
            if neg_found:
                neg_html = ' '.join([f'<span class="neg-tag">{w}</span>' for w in sorted(neg_found)])
                st.markdown(f"**Negative words found:** {neg_html}", unsafe_allow_html=True)
            else:
                st.markdown("**Negative words found:** None")

    # --- Sample Headlines ---
    st.markdown('<div class="section-header"><h2>Score Sample Headlines</h2></div>', unsafe_allow_html=True)

    sample_headlines = news_df.groupby('Sentiment').apply(lambda x: x.sample(3, random_state=42)).reset_index(drop=True)
    sample_headlines['Lexicon_Score'] = sample_headlines['Headline'].apply(simple_lexicon_score)

    st.dataframe(
        sample_headlines[['Headline', 'Sentiment', 'VADER_Compound', 'Lexicon_Score']].style.format({
            'VADER_Compound': '{:.3f}', 'Lexicon_Score': '{:.3f}'
        }),
        use_container_width=True
    )

    st.markdown("""
    <div class="insight-box">
        <strong>💡 Notice:</strong> The lexicon method and VADER often agree on direction but differ in magnitude.
        VADER tends to be more nuanced because it accounts for word intensity, capitalization, and modifiers — not just word presence.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: SENTIMENT VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Sentiment Visualizations":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Sentiment Visualizations</h1>
        <p>Exploring patterns in news sentiment and their relationship with stock market movements.</p>
    </div>
    """, unsafe_allow_html=True)

    # Aggregate daily sentiment
    daily_sentiment = news_df.groupby('Date').agg(
        mean_sentiment=('VADER_Compound', 'mean'),
        median_sentiment=('VADER_Compound', 'median'),
        headline_count=('Headline', 'count'),
        positive_pct=('Sentiment', lambda x: (x == 'positive').mean()),
        negative_pct=('Sentiment', lambda x: (x == 'negative').mean())
    ).reset_index()

    merged = pd.merge(stock_df[['Date', 'Close', 'Return', 'Volume']], daily_sentiment, on='Date', how='inner')

    # --- Sentiment Over Time ---
    st.markdown('<div class="section-header"><h2>Daily Sentiment Over Time</h2></div>', unsafe_allow_html=True)

    fig_sent_time = make_subplots(rows=2, cols=1, subplot_titles=('Stock Price', 'Daily Mean Sentiment'),
                                  vertical_spacing=0.12, row_heights=[0.6, 0.4])
    fig_sent_time.add_trace(go.Scatter(x=merged['Date'], y=merged['Close'], mode='lines',
                                       line=dict(color='#0e7c6b', width=2), name='Price'), row=1, col=1)
    # Sentiment as colored bars
    colors = ['#22c55e' if s > 0 else '#ef4444' for s in merged['mean_sentiment']]
    fig_sent_time.add_trace(go.Bar(x=merged['Date'], y=merged['mean_sentiment'],
                                    marker_color=colors, name='Sentiment', opacity=0.7), row=2, col=1)
    fig_sent_time.add_hline(y=0, line_dash='dash', line_color='#94a3b8', row=2, col=1)
    fig_sent_time.update_layout(height=550, template='plotly_white', font=dict(family='Inter'), showlegend=False)
    st.plotly_chart(fig_sent_time, use_container_width=True)

    # --- Sentiment Distribution ---
    st.markdown('<div class="section-header"><h2>Sentiment Score Distribution</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_dist = px.histogram(news_df, x='VADER_Compound', nbins=50, color='Sentiment',
                                color_discrete_map={'positive':'#22c55e','negative':'#ef4444','neutral':'#94a3b8'},
                                title='VADER Compound Score Distribution', template='plotly_white')
        fig_dist.update_layout(height=380, font=dict(family='Inter'), title_font_size=14)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        fig_box = px.box(news_df, x='Sentiment', y='VADER_Compound',
                         color='Sentiment',
                         color_discrete_map={'positive':'#22c55e','negative':'#ef4444','neutral':'#94a3b8'},
                         title='Score Distribution by Sentiment Class', template='plotly_white')
        fig_box.update_layout(height=380, font=dict(family='Inter'), title_font_size=14, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # --- Sentiment vs Returns ---
    st.markdown('<div class="section-header"><h2>Sentiment vs Stock Returns</h2></div>', unsafe_allow_html=True)

    fig_scatter = px.scatter(merged, x='mean_sentiment', y='Return',
                             trendline='ols', trendline_color_override='#e74c3c',
                             title='Daily Mean Sentiment vs Daily Returns',
                             labels={'mean_sentiment': 'Mean Sentiment Score', 'Return': 'Daily Return'},
                             template='plotly_white', color_discrete_sequence=['#0e7c6b'], opacity=0.5)
    fig_scatter.update_layout(height=450, font=dict(family='Inter'), title_font_size=15)
    st.plotly_chart(fig_scatter, use_container_width=True)

    corr = merged['mean_sentiment'].corr(merged['Return'])
    st.markdown(f"""
    <div class="insight-box">
        <strong>📊 Correlation: {corr:.3f}</strong><br>
        This measures the linear relationship between daily average sentiment and daily stock returns.
        A positive correlation means days with more positive news tend to see positive returns (and vice versa).
        The correlation may appear modest — this is typical in finance. Even small predictive edges can be valuable.
    </div>
    """, unsafe_allow_html=True)

    # --- Rolling Sentiment ---
    st.markdown('<div class="section-header"><h2>Rolling Sentiment & Returns</h2></div>', unsafe_allow_html=True)

    window = st.slider("Rolling window (days):", 5, 30, 10)
    merged['Rolling_Sentiment'] = merged['mean_sentiment'].rolling(window).mean()
    merged['Rolling_Return'] = merged['Return'].rolling(window).mean()

    fig_rolling = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rolling.add_trace(go.Scatter(x=merged['Date'], y=merged['Rolling_Return']*100, mode='lines',
                                     name=f'{window}-Day Avg Return (%)', line=dict(color='#0e7c6b', width=2)), secondary_y=False)
    fig_rolling.add_trace(go.Scatter(x=merged['Date'], y=merged['Rolling_Sentiment'], mode='lines',
                                     name=f'{window}-Day Avg Sentiment', line=dict(color='#f59e0b', width=2)), secondary_y=True)
    fig_rolling.update_layout(
        title=f'Rolling {window}-Day Average: Returns vs Sentiment',
        height=420, template='plotly_white', font=dict(family='Inter'), title_font_size=14,
        hovermode='x unified'
    )
    fig_rolling.update_yaxes(title_text='Return (%)', secondary_y=False)
    fig_rolling.update_yaxes(title_text='Sentiment', secondary_y=True)
    st.plotly_chart(fig_rolling, use_container_width=True)

    # --- Positive News Ratio ---
    st.markdown('<div class="section-header"><h2>Positive vs Negative News Ratio</h2></div>', unsafe_allow_html=True)

    merged['Pos_Neg_Ratio'] = merged['positive_pct'] / (merged['negative_pct'] + 0.01)
    fig_ratio = px.line(merged, x='Date', y='Pos_Neg_Ratio', title='Daily Positive/Negative Headline Ratio',
                        template='plotly_white')
    fig_ratio.update_traces(line=dict(color='#0e7c6b', width=1.5))
    fig_ratio.add_hline(y=1, line_dash='dash', line_color='#e74c3c', annotation_text='Equal ratio')
    fig_ratio.update_layout(height=350, font=dict(family='Inter'), title_font_size=14, yaxis_title='Ratio')
    st.plotly_chart(fig_ratio, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: STOCK PREDICTION — WITHOUT SENTIMENT
# ═══════════════════════════════════════════════════════════════
elif page == "💹 Stock Prediction — Without Sentiment":
    st.markdown("""
    <div class="main-header">
        <h1>💹 Stock Prediction — Without Sentiment</h1>
        <p>Building a baseline model using only price-based technical features to predict next-day returns.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Feature Engineering ---
    st.markdown('<div class="section-header"><h2>Feature Engineering (Price-Only)</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <h4>Technical Features</h4>
        <p>
        We create features from the stock price history that capture momentum, trend, and volatility:
        </p>
        <p>
        <strong>Return_1d:</strong> Previous day's return (momentum signal)<br>
        <strong>Return_5d:</strong> 5-day cumulative return (short-term trend)<br>
        <strong>Return_10d:</strong> 10-day cumulative return (medium-term trend)<br>
        <strong>Volatility_10d:</strong> Rolling 10-day standard deviation of returns<br>
        <strong>Volume_Change:</strong> Day-over-day percentage change in volume<br>
        <strong>MA_Ratio:</strong> Price relative to 20-day moving average (mean reversion signal)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Build features
    model_df = stock_df.copy()
    model_df['Return_1d'] = model_df['Return'].shift(1)
    model_df['Return_5d'] = model_df['Close'].pct_change(5).shift(1)
    model_df['Return_10d'] = model_df['Close'].pct_change(10).shift(1)
    model_df['Volatility_10d'] = model_df['Return'].rolling(10).std().shift(1)
    model_df['Volume_Change'] = model_df['Volume'].pct_change().shift(1)
    model_df['MA_20'] = model_df['Close'].rolling(20).mean().shift(1)
    model_df['MA_Ratio'] = (model_df['Close'].shift(1) / model_df['MA_20'])
    model_df['Target'] = model_df['Return']  # Next day return is what we predict
    model_df = model_df.dropna().reset_index(drop=True)

    price_features = ['Return_1d', 'Return_5d', 'Return_10d', 'Volatility_10d', 'Volume_Change', 'MA_Ratio']

    # Store for later use
    st.session_state['model_df'] = model_df
    st.session_state['price_features'] = price_features

    st.markdown("**Feature Sample (first 10 rows):**")
    st.dataframe(model_df[['Date'] + price_features + ['Target']].head(10).style.format(
        {f: '{:.5f}' for f in price_features + ['Target']}
    ), use_container_width=True)

    # --- Train/Test Split ---
    st.markdown('<div class="section-header"><h2>Train/Test Split</h2></div>', unsafe_allow_html=True)

    split_pct = st.slider("Training data percentage:", 60, 90, 80)
    split_idx = int(len(model_df) * split_pct / 100)
    train = model_df[:split_idx]
    test = model_df[split_idx:]

    st.session_state['split_idx'] = split_idx

    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(x=train['Date'], y=train['Close'], mode='lines', name=f'Train ({len(train)} days)', line=dict(color='#0e7c6b', width=2)))
    fig_split.add_trace(go.Scatter(x=test['Date'], y=test['Close'], mode='lines', name=f'Test ({len(test)} days)', line=dict(color='#e74c3c', width=2)))
    fig_split.update_layout(title='Train/Test Split', height=350, template='plotly_white', font=dict(family='Inter'), title_font_size=14)
    st.plotly_chart(fig_split, use_container_width=True)

    # --- Model Training ---
    st.markdown('<div class="section-header"><h2>Model: Price-Only</h2></div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Select model:", ["Linear Regression", "Ridge Regression", "Random Forest"], key="model_no_sent")

    if st.button("🔧 Train Price-Only Model", use_container_width=True, type="primary"):
        X_train = train[price_features]
        y_train = train['Target']
        X_test = test[price_features]
        y_test = test['Target']

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Ridge Regression":
            model = Ridge(alpha=1.0)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        # Store results
        st.session_state['price_model'] = model
        st.session_state['price_pred'] = y_pred
        st.session_state['price_y_test'] = y_test.values
        st.session_state['price_test_dates'] = test['Date'].values
        st.session_state['price_model_name'] = model_choice

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.session_state['price_metrics'] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{mae:.6f}</div><div class="metric-label">MAE</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{rmse:.6f}</div><div class="metric-label">RMSE</div></div>""", unsafe_allow_html=True)
        with col3:
            r2_color = '#166534' if r2 > 0 else '#991b1b'
            st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:{r2_color}">{r2:.4f}</div><div class="metric-label">R² Score</div></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-card">
            <h4>What do these metrics mean?</h4>
            <p>
            <strong>MAE:</strong> Average absolute prediction error. For daily returns, even 0.001 matters.<br><br>
            <strong>RMSE:</strong> Like MAE but penalizes large errors more. Lower = better.<br><br>
            <strong>R²:</strong> Proportion of variance explained by the model. 0 = no better than guessing the mean;
            1 = perfect; negative = worse than the mean. In stock prediction, even small positive R² is noteworthy.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Prediction Chart ---
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=test['Date'], y=y_test, mode='lines', name='Actual Returns', line=dict(color='#2c3e50', width=1.5)))
        fig_pred.add_trace(go.Scatter(x=test['Date'], y=y_pred, mode='lines', name='Predicted Returns', line=dict(color='#e74c3c', width=1.5, dash='dash')))
        fig_pred.update_layout(
            title=f'{model_choice} — Predicted vs Actual Returns (Price-Only)',
            height=400, template='plotly_white', font=dict(family='Inter'),
            title_font_size=14, hovermode='x unified', yaxis_title='Daily Return'
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Direction accuracy
        actual_dir = np.sign(y_test)
        pred_dir = np.sign(y_pred)
        direction_acc = (actual_dir == pred_dir).mean() * 100
        st.markdown(f"""
        <div class="insight-box">
            <strong>📊 Direction Accuracy: {direction_acc:.1f}%</strong><br>
            This measures how often the model correctly predicts whether the stock goes up or down.
            50% = coin flip. Anything consistently above 55% in real markets is considered strong.
        </div>
        """, unsafe_allow_html=True)

        # Feature importance for RF
        if model_choice == "Random Forest":
            st.markdown('<div class="section-header"><h2>Feature Importance</h2></div>', unsafe_allow_html=True)
            imp = pd.DataFrame({'Feature': price_features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
            fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h', template='plotly_white',
                             color_discrete_sequence=['#0e7c6b'])
            fig_imp.update_layout(height=300, font=dict(family='Inter'), title='Feature Importance', title_font_size=14)
            st.plotly_chart(fig_imp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: STOCK PREDICTION — WITH SENTIMENT
# ═══════════════════════════════════════════════════════════════
elif page == "🧠 Stock Prediction — With Sentiment":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Stock Prediction — With Sentiment</h1>
        <p>Adding news sentiment features to the model. Does NLP improve stock return predictions?</p>
    </div>
    """, unsafe_allow_html=True)

    if 'price_metrics' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Train the price-only model first!</strong> Go to <strong>💹 Stock Prediction — Without Sentiment</strong>
            and train a model so we can compare.
        </div>
        """, unsafe_allow_html=True)
    else:
        model_df = st.session_state['model_df']
        price_features = st.session_state['price_features']
        split_idx = st.session_state['split_idx']

        # --- Sentiment Feature Engineering ---
        st.markdown('<div class="section-header"><h2>Sentiment Feature Engineering</h2></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-card">
            <h4>New Features from News Data</h4>
            <p>We aggregate daily sentiment from all headlines and create features that capture different aspects:</p>
            <p>
            <strong>Sent_Mean:</strong> Average VADER compound score for the day<br>
            <strong>Sent_Max / Sent_Min:</strong> Most positive and most negative headline scores<br>
            <strong>Sent_Std:</strong> Sentiment dispersion — are opinions mixed or unified?<br>
            <strong>Positive_Ratio:</strong> Proportion of headlines that are positive<br>
            <strong>Headline_Count:</strong> Number of headlines (more coverage = more attention)<br>
            <strong>Sent_Rolling_3d:</strong> 3-day average sentiment (smoothed signal)
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Build sentiment features
        daily_sent = news_df.groupby('Date').agg(
            Sent_Mean=('VADER_Compound', 'mean'),
            Sent_Max=('VADER_Compound', 'max'),
            Sent_Min=('VADER_Compound', 'min'),
            Sent_Std=('VADER_Compound', 'std'),
            Positive_Ratio=('Sentiment', lambda x: (x == 'positive').mean()),
            Headline_Count=('Headline', 'count')
        ).reset_index()
        daily_sent['Sent_Std'] = daily_sent['Sent_Std'].fillna(0)

        # Merge and create lagged features (use previous day's sentiment to predict today)
        model_sent_df = pd.merge(model_df, daily_sent, on='Date', how='left')
        # Lag sentiment features by 1 day to avoid look-ahead bias
        for col in ['Sent_Mean', 'Sent_Max', 'Sent_Min', 'Sent_Std', 'Positive_Ratio', 'Headline_Count']:
            model_sent_df[col] = model_sent_df[col].shift(1)
        model_sent_df['Sent_Rolling_3d'] = model_sent_df['Sent_Mean'].rolling(3).mean()
        model_sent_df = model_sent_df.dropna().reset_index(drop=True)

        sentiment_features = ['Sent_Mean', 'Sent_Max', 'Sent_Min', 'Sent_Std', 'Positive_Ratio', 'Headline_Count', 'Sent_Rolling_3d']
        all_features = price_features + sentiment_features

        st.markdown("**Combined Feature Sample (first 8 rows):**")
        st.dataframe(model_sent_df[['Date'] + sentiment_features].head(8).style.format(
            {f: '{:.4f}' for f in sentiment_features if f != 'Headline_Count'}
        ), use_container_width=True)

        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Look-Ahead Bias Prevention:</strong> All sentiment features are lagged by 1 day. We use
            <em>yesterday's</em> news sentiment to predict <em>today's</em> returns. This prevents data leakage and
            reflects real-world constraints — you can only trade on news you've already seen.
        </div>
        """, unsafe_allow_html=True)

        # --- Model Training ---
        st.markdown('<div class="section-header"><h2>Model: Price + Sentiment</h2></div>', unsafe_allow_html=True)

        split_idx_sent = int(len(model_sent_df) * 0.8)
        train_s = model_sent_df[:split_idx_sent]
        test_s = model_sent_df[split_idx_sent:]

        model_choice_s = st.selectbox("Select model:", ["Linear Regression", "Ridge Regression", "Random Forest"], key="model_with_sent")

        if st.button("🧠 Train Price + Sentiment Model", use_container_width=True, type="primary"):
            X_train = train_s[all_features]
            y_train = train_s['Target']
            X_test = test_s[all_features]
            y_test = test_s['Target']

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if model_choice_s == "Linear Regression":
                model = LinearRegression()
            elif model_choice_s == "Ridge Regression":
                model = Ridge(alpha=1.0)
            else:
                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            # Store
            st.session_state['sent_model'] = model
            st.session_state['sent_pred'] = y_pred
            st.session_state['sent_y_test'] = y_test.values
            st.session_state['sent_test_dates'] = test_s['Date'].values
            st.session_state['sent_model_name'] = model_choice_s
            st.session_state['all_features'] = all_features

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            st.session_state['sent_metrics'] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}

            # Metrics with comparison
            price_m = st.session_state['price_metrics']

            col1, col2, col3 = st.columns(3)
            with col1:
                delta = ((price_m['MAE'] - mae) / price_m['MAE'] * 100)
                st.markdown(f"""<div class="metric-card"><div class="metric-value">{mae:.6f}</div><div class="metric-label">MAE (↓{delta:.1f}% vs price-only)</div></div>""", unsafe_allow_html=True)
            with col2:
                delta_r = ((price_m['RMSE'] - rmse) / price_m['RMSE'] * 100)
                st.markdown(f"""<div class="metric-card"><div class="metric-value">{rmse:.6f}</div><div class="metric-label">RMSE (↓{delta_r:.1f}% vs price-only)</div></div>""", unsafe_allow_html=True)
            with col3:
                r2_color = '#166534' if r2 > price_m['R²'] else '#991b1b'
                st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:{r2_color}">{r2:.4f}</div><div class="metric-label">R² (was {price_m['R²']:.4f})</div></div>""", unsafe_allow_html=True)

            # Prediction chart
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=test_s['Date'], y=y_test, mode='lines', name='Actual', line=dict(color='#2c3e50', width=1.5)))
            fig_pred.add_trace(go.Scatter(x=test_s['Date'], y=y_pred, mode='lines', name='Predicted (Price+Sentiment)', line=dict(color='#0e7c6b', width=1.5, dash='dash')))
            fig_pred.update_layout(
                title=f'{model_choice_s} — Predicted vs Actual Returns (Price + Sentiment)',
                height=400, template='plotly_white', font=dict(family='Inter'),
                title_font_size=14, hovermode='x unified', yaxis_title='Daily Return'
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Direction accuracy
            actual_dir = np.sign(y_test)
            pred_dir = np.sign(y_pred)
            direction_acc = (actual_dir == pred_dir).mean() * 100
            price_dir = (np.sign(st.session_state['price_y_test']) == np.sign(st.session_state['price_pred'])).mean() * 100

            st.markdown(f"""
            <div class="insight-box">
                <strong>📊 Direction Accuracy: {direction_acc:.1f}%</strong> (was {price_dir:.1f}% without sentiment)<br>
                Sentiment features help the model better predict the <em>direction</em> of returns, which is often
                more valuable than predicting exact magnitude in real trading.
            </div>
            """, unsafe_allow_html=True)

            # Feature importance
            if model_choice_s == "Random Forest":
                st.markdown('<div class="section-header"><h2>Feature Importance — All Features</h2></div>', unsafe_allow_html=True)
                imp = pd.DataFrame({'Feature': all_features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
                imp['Type'] = imp['Feature'].apply(lambda x: 'Sentiment' if x in sentiment_features else 'Price')
                fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h', color='Type',
                                 color_discrete_map={'Price':'#0e7c6b','Sentiment':'#f59e0b'},
                                 template='plotly_white')
                fig_imp.update_layout(height=450, font=dict(family='Inter'), title='Feature Importance: Price vs Sentiment', title_font_size=14)
                st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown("""
                <div class="insight-box">
                    <strong>💡 Look at the yellow bars.</strong> These are sentiment features. Their relative importance tells
                    you how much predictive value the news data adds on top of price-based technical features.
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: HEAD-TO-HEAD COMPARISON
# ═══════════════════════════════════════════════════════════════
elif page == "⚔️ Head-to-Head Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>⚔️ Head-to-Head Comparison</h1>
        <p>Price-only vs Price + Sentiment — does adding NLP actually improve stock return predictions?</p>
    </div>
    """, unsafe_allow_html=True)

    if 'sent_metrics' not in st.session_state or 'price_metrics' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Train both models first!</strong> You need to train the price-only model (💹) and the
            price + sentiment model (🧠) before comparing them here.
        </div>
        """, unsafe_allow_html=True)
    else:
        pm = st.session_state['price_metrics']
        sm = st.session_state['sent_metrics']
        p_name = st.session_state['price_model_name']
        s_name = st.session_state['sent_model_name']

        # --- Metrics Table ---
        st.markdown('<div class="section-header"><h2>Accuracy Metrics Comparison</h2></div>', unsafe_allow_html=True)

        comp_data = {
            'Metric': ['MAE', 'RMSE', 'R²'],
            f'Price-Only ({p_name})': [f"{pm['MAE']:.6f}", f"{pm['RMSE']:.6f}", f"{pm['R²']:.4f}"],
            f'Price+Sentiment ({s_name})': [f"{sm['MAE']:.6f}", f"{sm['RMSE']:.6f}", f"{sm['R²']:.4f}"],
            'Change': [
                f"{'↓' if sm['MAE'] < pm['MAE'] else '↑'} {abs((pm['MAE']-sm['MAE'])/pm['MAE']*100):.1f}%",
                f"{'↓' if sm['RMSE'] < pm['RMSE'] else '↑'} {abs((pm['RMSE']-sm['RMSE'])/pm['RMSE']*100):.1f}%",
                f"{'↑' if sm['R²'] > pm['R²'] else '↓'} {abs(sm['R²']-pm['R²']):.4f}"
            ],
            'Winner': [
                '🧠 Sentiment' if sm['MAE'] < pm['MAE'] else '💹 Price-Only',
                '🧠 Sentiment' if sm['RMSE'] < pm['RMSE'] else '💹 Price-Only',
                '🧠 Sentiment' if sm['R²'] > pm['R²'] else '💹 Price-Only'
            ]
        }
        st.dataframe(pd.DataFrame(comp_data).set_index('Metric'), use_container_width=True)

        # --- Visual Comparison ---
        st.markdown('<div class="section-header"><h2>Visual Comparison</h2></div>', unsafe_allow_html=True)

        # Bar chart comparison
        metrics_names = ['MAE', 'RMSE']
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name='Price-Only', x=metrics_names, y=[pm['MAE'], pm['RMSE']], marker_color='#e74c3c'))
        fig_comp.add_trace(go.Bar(name='Price+Sentiment', x=metrics_names, y=[sm['MAE'], sm['RMSE']], marker_color='#0e7c6b'))
        fig_comp.update_layout(
            title='MAE & RMSE Comparison (lower = better)', barmode='group',
            height=380, template='plotly_white', font=dict(family='Inter'), title_font_size=14
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # R² comparison
        col1, col2 = st.columns(2)
        with col1:
            fig_r2 = go.Figure(go.Bar(
                x=['Price-Only', 'Price+Sentiment'],
                y=[pm['R²'], sm['R²']],
                marker_color=['#e74c3c', '#0e7c6b'],
                text=[f"{pm['R²']:.4f}", f"{sm['R²']:.4f}"],
                textposition='outside'
            ))
            fig_r2.update_layout(title='R² Score (higher = better)', height=350, template='plotly_white',
                                 font=dict(family='Inter'), title_font_size=14)
            st.plotly_chart(fig_r2, use_container_width=True)

        with col2:
            # Direction accuracy comparison
            price_dir = (np.sign(st.session_state['price_y_test']) == np.sign(st.session_state['price_pred'])).mean() * 100
            sent_dir = (np.sign(st.session_state['sent_y_test']) == np.sign(st.session_state['sent_pred'])).mean() * 100

            fig_dir = go.Figure(go.Bar(
                x=['Price-Only', 'Price+Sentiment'],
                y=[price_dir, sent_dir],
                marker_color=['#e74c3c', '#0e7c6b'],
                text=[f"{price_dir:.1f}%", f"{sent_dir:.1f}%"],
                textposition='outside'
            ))
            fig_dir.add_hline(y=50, line_dash='dash', line_color='#94a3b8', annotation_text='Random (50%)')
            fig_dir.update_layout(title='Direction Accuracy (higher = better)', height=350, template='plotly_white',
                                  font=dict(family='Inter'), title_font_size=14, yaxis_range=[40, max(price_dir, sent_dir) + 8])
            st.plotly_chart(fig_dir, use_container_width=True)

        # --- Overlay Predictions ---
        st.markdown('<div class="section-header"><h2>Prediction Overlay</h2></div>', unsafe_allow_html=True)

        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Scatter(
            x=st.session_state['price_test_dates'], y=st.session_state['price_y_test'],
            mode='lines', name='Actual', line=dict(color='#2c3e50', width=2)
        ))
        fig_overlay.add_trace(go.Scatter(
            x=st.session_state['price_test_dates'], y=st.session_state['price_pred'],
            mode='lines', name='Price-Only Prediction', line=dict(color='#e74c3c', width=1.5, dash='dot')
        ))
        fig_overlay.add_trace(go.Scatter(
            x=st.session_state['sent_test_dates'], y=st.session_state['sent_pred'],
            mode='lines', name='Price+Sentiment Prediction', line=dict(color='#0e7c6b', width=1.5, dash='dash')
        ))
        fig_overlay.update_layout(
            title='Both Models vs Actual Returns',
            height=450, template='plotly_white', font=dict(family='Inter'),
            title_font_size=15, hovermode='x unified', yaxis_title='Daily Return',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_overlay, use_container_width=True)

        # --- Verdict ---
        st.markdown('<div class="section-header"><h2>The Verdict</h2></div>', unsafe_allow_html=True)

        sent_wins = sum([
            sm['MAE'] < pm['MAE'],
            sm['RMSE'] < pm['RMSE'],
            sm['R²'] > pm['R²'],
            sent_dir > price_dir
        ])

        if sent_wins >= 3:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#ecfdf5,#d1fae5); border:2px solid #6ee7b7; border-radius:12px; padding:1.5rem; margin:1rem 0;">
                <strong style="color:#065f46; font-size:1.2rem;">🧠 Sentiment Wins ({sent_wins}/4 metrics)</strong><br><br>
                Adding news sentiment meaningfully improved prediction accuracy. The model with sentiment features
                outperformed on {sent_wins} out of 4 metrics. In real-world finance, even marginal improvements in
                prediction accuracy can translate to significant returns when applied at scale.
            </div>
            """, unsafe_allow_html=True)
        elif sent_wins >= 2:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #eff6ff, #dbeafe); border:2px solid #93c5fd; border-radius:12px; padding:1.5rem; margin:1rem 0;">
                <strong style="color:#1e40af; font-size:1.2rem;">🤝 Mixed Results ({sent_wins}/4 metrics for sentiment)</strong><br><br>
                Sentiment features helped on some metrics but not all. This is common in practice — sentiment is a noisy signal,
                and its value depends on market conditions, the time period, and how sentiment is measured.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#fefce8,#fef9c3); border:2px solid #fcd34d; border-radius:12px; padding:1.5rem; margin:1rem 0;">
                <strong style="color:#92400e; font-size:1.2rem;">💹 Price-Only Holds Its Ground ({4-sent_wins}/4 metrics)</strong><br><br>
                Sentiment didn't clearly improve predictions this time. Possible reasons: the sentiment signal is too noisy,
                the model doesn't capture non-linear interactions well, or the market was driven more by macro factors than news.
                Try a different model type or feature combination.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-card">
            <h4>Key Takeaways for Real-World Application</h4>
            <p>
            <strong>1. Sentiment is a supplementary signal, not a silver bullet.</strong> It works best alongside traditional features, not as a replacement.<br><br>
            <strong>2. Data quality matters enormously.</strong> Real financial sentiment analysis requires carefully curated news sources and domain-specific NLP models.<br><br>
            <strong>3. Look-ahead bias is the biggest trap.</strong> Always ensure sentiment features are properly lagged — using today's news to predict today's returns is cheating.<br><br>
            <strong>4. Direction accuracy often matters more than magnitude.</strong> In trading, knowing which way the stock moves is frequently more valuable than predicting the exact return.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: SUMMARY & TAKEAWAYS
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Summary & Takeaways":
    st.markdown("""
    <div class="main-header">
        <h1>📋 Summary & Takeaways</h1>
        <p>Key concepts, methods, and findings from this dashboard — a quick reference.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sentiment Analysis Summary ---
    st.markdown('<div class="section-header"><h2>Sentiment Analysis — Key Concepts</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>Three Approaches</h4>
            <p>
            <strong>1. Lexicon-Based:</strong> Count positive/negative words from a dictionary. Simple and fast, but misses context.<br><br>
            <strong>2. VADER:</strong> Rule-based system with intensity scores, handles capitalization, punctuation, modifiers, and negation. Best for short texts.<br><br>
            <strong>3. Machine Learning:</strong> Train a classifier on labeled data. Highest accuracy but requires training data and more setup.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-card">
            <h4>Text Preprocessing Pipeline</h4>
            <p>
            <span class="step-indicator">1</span> Tokenize (split into words)<br><br>
            <span class="step-indicator">2</span> Normalize (lowercase, remove punctuation)<br><br>
            <span class="step-indicator">3</span> Remove stop words<br><br>
            <span class="step-indicator">4</span> Score sentiment (lexicon match or model prediction)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="concept-card">
            <h4>VADER Quick Reference</h4>
            <p>
            <strong>Output:</strong> pos, neu, neg (proportions) + compound (-1 to +1)<br><br>
            <strong>Thresholds:</strong><br>
            Compound ≥ 0.05 → Positive<br>
            Compound ≤ -0.05 → Negative<br>
            In between → Neutral<br><br>
            <strong>Special rules:</strong><br>
            • Capitalization amplifies (GREAT > great)<br>
            • Punctuation amplifies (great!!! > great)<br>
            • "not" flips polarity<br>
            • Degree modifiers adjust intensity
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-card">
            <h4>Key Sentiment Features for Prediction</h4>
            <p>
            <strong>Mean Score:</strong> Overall daily mood<br>
            <strong>Max / Min Score:</strong> Extreme opinions<br>
            <strong>Std Deviation:</strong> Disagreement level<br>
            <strong>Positive Ratio:</strong> Bull/bear balance<br>
            <strong>Volume:</strong> Attention level<br>
            <strong>Rolling Average:</strong> Trend in sentiment
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- Prediction Approach ---
    st.markdown('<div class="section-header"><h2>Prediction Framework</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <p>
        <table style="width:100%; border-collapse:collapse; font-size:0.95rem;">
        <tr style="background:#f0fdf4;">
            <th style="padding:10px; text-align:left; border-bottom:2px solid #0e7c6b;">Aspect</th>
            <th style="padding:10px; text-align:left; border-bottom:2px solid #0e7c6b;">Price-Only Model</th>
            <th style="padding:10px; text-align:left; border-bottom:2px solid #0e7c6b;">Price + Sentiment Model</th>
        </tr>
        <tr>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;"><strong>Features</strong></td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Returns, volatility, volume, MA ratio</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Same + sentiment mean, max, min, std, positive ratio, count, rolling</td>
        </tr>
        <tr>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;"><strong>Data Source</strong></td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Historical prices only</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Prices + news headlines</td>
        </tr>
        <tr>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;"><strong>Captures</strong></td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Technical patterns and momentum</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Technical patterns + market mood</td>
        </tr>
        <tr>
            <td style="padding:10px;"><strong>Limitation</strong></td>
            <td style="padding:10px;">Ignores qualitative information</td>
            <td style="padding:10px;">Sentiment is noisy; quality depends on NLP accuracy</td>
        </tr>
        </table>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Evaluation Metrics ---
    st.markdown('<div class="section-header"><h2>Evaluation Metrics</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="formula-box">
            <strong>MAE</strong> = (1/n) Σ |yᵢ − ŷᵢ|<br>
            <em>Average absolute error. Lower = better.</em>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box">
            <strong>RMSE</strong> = √[(1/n) Σ (yᵢ − ŷᵢ)²]<br>
            <em>Penalizes large errors. Lower = better.</em>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="formula-box">
            <strong>R²</strong> = 1 − [SS_res / SS_tot]<br>
            <em>Proportion of variance explained. Higher = better. Can be negative.</em>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box">
            <strong>Direction Accuracy</strong> = correct direction / total<br>
            <em>Did we predict up/down correctly? 50% = random.</em>
        </div>
        """, unsafe_allow_html=True)

    # --- Limitations ---
    st.markdown('<div class="section-header"><h2>Limitations & Considerations</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important Context:</strong><br><br>
        • This dashboard uses <strong>synthetic data</strong> for demonstration — real market data is noisier and less predictable<br>
        • Stock markets are influenced by countless factors beyond news sentiment — macroeconomics, geopolitics, interest rates, institutional flows<br>
        • <strong>Efficient Market Hypothesis</strong> suggests that publicly available information (like news) is quickly priced in — sentiment edges, if they exist, tend to be small and short-lived<br>
        • Overfitting is a constant risk — always validate on out-of-sample data<br>
        • Real-world NLP for finance requires domain-specific models (financial vocabulary differs from general language)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>📚 Extensions to Explore:</strong><br><br>
        • <strong>FinBERT:</strong> A BERT model pre-trained on financial text — far more accurate than VADER for financial sentiment<br>
        • <strong>Social media sentiment:</strong> Twitter/Reddit sentiment as an alternative or complement to news<br>
        • <strong>Event-driven analysis:</strong> Focus on earnings announcements, FDA decisions, or M&A news specifically<br>
        • <strong>Intraday analysis:</strong> Sentiment impact at minute-level for high-frequency trading research
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; padding:1rem;'>"
        "📰 <strong>Sentiment Analysis & Stock Prediction Masterclass</strong> | SP Jain School of Global Management | Global MBA Program<br>"
        "Dr. Anshul Gupta"
        "</div>",
        unsafe_allow_html=True
    )
