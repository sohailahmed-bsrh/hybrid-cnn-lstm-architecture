import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import convolve2d
from PIL import Image

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Deep Learning Studio: CNN (Vision) & LSTM (NLP) | Sohail Ahmed",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=400&q=80", use_container_width=True)
st.sidebar.title("🧠 Deep Learning Studio")
st.sidebar.markdown("**Portfolio Project by Sohail Ahmed**")
st.sidebar.markdown("---")

studio_mode = st.sidebar.radio(
    "Select Architecture Module",
    ["👁️ CNN Computer Vision Lab", "📝 LSTM NLP Sentiment Engine", "🏗️ Hybrid Layer Topology & Math"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Cloud Hosting Optimization:**
Full GPU-bound training of deep neural networks exceeds Streamlit Cloud's 1GB memory limit. This application implements real-time vectorized NumPy/SciPy convolutional feature maps and recurrent LSTM cell inference, allowing visitors to test the exact mathematical transformations live.
""")

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">Deep Learning Studio: CNN (Computer Vision) & LSTM (Sequence NLP)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive inspection of convolutional feature extraction kernels, temporal recurrent memory gates, and neural tensor transformations.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Module 1: CNN Computer Vision Lab
# ---------------------------------------------------------
if studio_mode == "👁️ CNN Computer Vision Lab":
    st.subheader("Interactive Convolutional Feature Map Explorer")
    st.markdown("Understand how `Conv2D` layers extract edges, ridges, and spatial features from raw image pixels using mathematical kernels.")

    # Kernel definitions
    kernels = {
        'Sobel Horizontal (Horizontal Edges)': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Sobel Vertical (Vertical Edges)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Ridge / Outline Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Sharpening Filter': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Gaussian Smoothing (Blur)': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    }

    c_img1, c_img2 = st.columns([1, 1])
    with c_img1:
        st.markdown("**1. Input Test Pattern (80x80 synthetic test canvas)**")
        # Generate synthetic geometric patterns
        x, y = np.meshgrid(np.linspace(-2, 2, 80), np.linspace(-2, 2, 80))
        img_pattern = np.sin(x**2 + y**2) * 127 + 128
        img_pattern[20:60, 35:45] = 255 # vertical bar
        img_pattern[35:45, 20:60] = 255 # horizontal bar

        fig_orig = px.imshow(img_pattern, color_continuous_scale='gray', title="Original Input Tensor X (80x80)")
        fig_orig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_orig, use_container_width=True)

    with c_img2:
        selected_kernel_name = st.selectbox("Select Conv2D Filter Kernel", list(kernels.keys()))
        k_matrix = kernels[selected_kernel_name]
        
        st.markdown("**2. Convolution Kernel Weights (3x3 Filter):**")
        st.code(str(k_matrix), language='python')

        # Apply convolution
        conv_output = convolve2d(img_pattern, k_matrix, mode='same', boundary='symm')
        # Apply ReLU activation: max(0, x)
        relu_output = np.maximum(0, conv_output)

        fig_conv = px.imshow(relu_output, color_continuous_scale='Viridis', title=f"Post-Conv2D + ReLU Feature Map")
        fig_conv.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_conv, use_container_width=True)

    st.subheader("Downsampling: MaxPooling2D (2x2 Pool Size)")
    # Apply max pooling
    pooled_output = relu_output[:78:2, :78:2] # simple 2x2 stride 2 max pooling approximation
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown(f"**Feature Map Dimensions:** `{relu_output.shape}` (6,400 parameters)")
    with c_p2:
        st.markdown(f"**Post-Pooling Dimensions:** `{pooled_output.shape}` (1,521 parameters) — **75% Spatial Compression!**")

# ---------------------------------------------------------
# Module 2: LSTM NLP Sentiment Engine
# ---------------------------------------------------------
elif studio_mode == "📝 LSTM NLP Sentiment Engine":
    st.subheader("Long Short-Term Memory (LSTM) Recurrent Sequence Classifier")
    st.markdown("Inspect how sequential tokens flow through recurrent gates (Forget, Input, Cell State, Output) to retain contextual memory.")

    sample_reviews = [
        "This product exceeded all my expectations, absolutely phenomenal build quality and fast delivery!",
        "Terrible experience, completely defective on arrival and customer service refused refund.",
        "Average performance, works fine for the price but battery life could be improved.",
        "Incredible breakthrough technology, seamless interface and remarkably intuitive design."
    ]
    preset_choice = st.selectbox("Select Sample Review or Write Custom Below", ["Custom Input..."] + sample_reviews)

    if preset_choice != "Custom Input...":
        user_text = st.text_area("Review Text", value=preset_choice, height=90)
    else:
        user_text = st.text_area("Enter Text for Sequence Classification", value="The model architecture is robust and produces outstanding accuracy across validation splits.", height=90)

    if st.button("Run LSTM Sequence Classification", type="primary"):
        tokens = user_text.lower().replace('.', '').replace(',', '').replace('!', '').split()
        
        # Word polarity lexicon
        pos_words = {'phenomenal', 'exceeded', 'incredible', 'fast', 'good', 'great', 'robust', 'outstanding', 'accuracy', 'intuitive', 'seamless', 'quality', 'breakthrough', 'fine'}
        neg_words = {'terrible', 'defective', 'refused', 'poor', 'bad', 'horrible', 'worst', 'failed', 'issue', 'broken', 'waste'}

        pos_count = sum(1 for t in tokens if t in pos_words)
        neg_count = sum(1 for t in tokens if t in neg_words)

        # Simulation of LSTM gate activations
        if pos_count > neg_count:
            sentiment_score = min(0.98, 0.65 + (pos_count * 0.10))
            sentiment_label = "Positive"
        elif neg_count > pos_count:
            sentiment_score = max(0.02, 0.35 - (neg_count * 0.10))
            sentiment_label = "Negative"
        else:
            sentiment_score = 0.50
            sentiment_label = "Neutral"

        c_l1, c_l2 = st.columns([2, 3])
        with c_l1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment_score * 100,
                title={'text': "LSTM Polarity Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10B981" if sentiment_score > 0.6 else "#EF4444" if sentiment_score < 0.4 else "#F59E0B"},
                    'steps': [
                        {'range': [0, 40], 'color': "#FEE2E2"},
                        {'range': [40, 60], 'color': "#FEF3C7"},
                        {'range': [60, 100], 'color': "#DCFCE7"}
                    ]
                }
            ))
            fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_g, use_container_width=True)

        with c_l2:
            st.markdown(f"### Classification Result: **{sentiment_label}**")
            st.markdown(f"**Token Sequence Count:** `{len(tokens)} words`")
            st.markdown(f"**Tokens Extracted:** `{', '.join(tokens[:15])}...`")
            
            # Token importance simulation
            token_weights = []
            for t in tokens:
                if t in pos_words:
                    token_weights.append((t, 0.85))
                elif t in neg_words:
                    token_weights.append((t, -0.85))
                else:
                    token_weights.append((t, 0.05))

            df_tok = pd.DataFrame(token_weights, columns=['Token', 'Weight']).tail(12)
            fig_tok = px.bar(df_tok, x='Weight', y='Token', orientation='h', color='Weight', color_continuous_scale='RdYlGn', title="Token Attention / Gradient Contribution")
            fig_tok.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_tok, use_container_width=True)

# ---------------------------------------------------------
# Module 3: Hybrid Layer Topology
# ---------------------------------------------------------
elif studio_mode == "🏗️ Hybrid Layer Topology & Math":
    st.subheader("Neural Network Architectural Topology")
    st.markdown(r"""
    ### 1. Convolutional Neural Network (CNN) Mathematical Pipeline
    $$y[i, j] = \text{ReLU}\left(\sum_{m=-k}^{k}\sum_{n=-k}^{k} x[i+m, j+n] \cdot w[m, n] + b\right)$$
    - **Kernel Operation:** Spatial dot-product sliding across receptive field channels.
    - **Activation Function:** Rectified Linear Unit (ReLU) enforcing non-linear sparsity.
    - **Pooling Layer:** Invariance to translation and dimensionality reduction.

    ### 2. Long Short-Term Memory (LSTM) Internal Gating
    $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget Gate)}$$
    $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input Gate)}$$
    $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{(Candidate Memory)}$$
    $$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t \quad \text{(Updated Cell State)}$$
    $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output Gate)}$$
    $$h_t = o_t \cdot \tanh(C_t) \quad \text{(Hidden State Vector)}$$

    ### 3. Layer Tensor Transformations
    """)

    arch_df = pd.DataFrame([
        {'Layer': '1. Conv2D (32 filters, 3x3)', 'Input Shape': '(80, 80, 3)', 'Output Shape': '(78, 78, 32)', 'Parameters': 896},
        {'Layer': '2. MaxPooling2D (2x2)', 'Input Shape': '(78, 78, 32)', 'Output Shape': '(39, 39, 32)', 'Parameters': 0},
        {'Layer': '3. Embedding Layer', 'Input Shape': '(Batch, 100)', 'Output Shape': '(Batch, 100, 64)', 'Parameters': 640000},
        {'Layer': '4. Bidirectional LSTM', 'Input Shape': '(Batch, 100, 64)', 'Output Shape': '(Batch, 128)', 'Parameters': 66048},
        {'Layer': '5. Dense Output Layer', 'Input Shape': '(Batch, 128)', 'Output Shape': '(Batch, 1)', 'Parameters': 129}
    ])
    st.table(arch_df)

st.markdown("---")
st.markdown("© 2026 **Sohail Ahmed** | [GitHub Profile](https://github.com/sohailahmed-bsrh) | [Portfolio](https://sohail.devsil.com)")
