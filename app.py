"""
Streamlit Web App for Bird/Plane/Superman Image Classifier

Run with: streamlit run app.py
"""

import streamlit as st
from PIL import Image
import tempfile
import os
from pathlib import Path
from classifier import BirdPlaneSupermanClassifier
import torch

# Page configuration
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🦅",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path, threshold):
    """Load and cache the classifier model"""
    return BirdPlaneSupermanClassifier(model_path, confidence_threshold=threshold)


def get_confidence_color(confidence):
    """Return CSS class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"


def main():
    # Header
    st.title("🦅 ✈️ 🦸 Image Classifier")
    st.markdown("Upload an image to classify it as **Bird**, **Plane**, **Superman**, or **Other**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/best_model.pth",
        help="Path to the trained model file"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence to predict main classes. Lower confidence → 'other'"
    )
    
    # Check if model exists
    if not Path(model_path).exists():
        st.error(f"❌ Model file not found at: {model_path}")
        st.info("Please train the model first or update the model path in the sidebar.")
        return
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            classifier = load_model(model_path, confidence_threshold)
        
        # Display model info in sidebar
        st.sidebar.success("✅ Model loaded!")
        model_info = classifier.get_model_info()
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Model Info")
        st.sidebar.write(f"**Device:** {model_info['device']}")
        st.sidebar.write(f"**Classes:** {', '.join(model_info['classes'])}")
        
        if 'config' in model_info and model_info['config']:
            config = model_info['config']
            if 'model' in config:
                st.sidebar.write(f"**Architecture:** {config['model']}")
            if 'val_accuracy' in config:
                st.sidebar.write(f"**Val Accuracy:** {config['val_accuracy']:.2%}")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Main content
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a JPG, JPEG, or PNG image"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Make prediction
        with col2:
            st.subheader("🎯 Prediction Results")
            
            # Save to temporary file for prediction
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_path = tmp_file.name
            
            try:
                with st.spinner("Analyzing image..."):
                    pred_class, confidence, all_probs = classifier.predict(temp_path)
                
                # Display main prediction
                confidence_class = get_confidence_color(confidence)
                
                st.markdown(f"""
                    <div class="prediction-box">
                        <p style="font-size: 18px; margin-bottom: 5px;">Predicted Class:</p>
                        <p class="big-font">{pred_class.upper()}</p>
                        <p style="font-size: 16px; margin-top: 10px;">Confidence:</p>
                        <p class="{confidence_class}" style="font-size: 28px;">{confidence:.1%}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display confidence interpretation
                if confidence >= 0.8:
                    st.success("🟢 High confidence prediction")
                elif confidence >= 0.5:
                    st.warning("🟡 Medium confidence prediction")
                else:
                    st.error("🔴 Low confidence prediction")
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        # Display all class probabilities
        st.markdown("---")
        st.subheader("📊 All Class Probabilities")
        
        # Sort probabilities by value (highest first)
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, probability in sorted_probs:
            # Add emoji for each class
            emoji_map = {
                'bird': '🦅',
                'plane': '✈️',
                'superman': '🦸',
                'other': '❓'
            }
            emoji = emoji_map.get(class_name, '📦')
            
            # Create progress bar
            st.write(f"{emoji} **{class_name.capitalize()}**")
            st.progress(probability)
            st.write(f"{probability:.2%}")
            st.markdown("")
        
        # Additional information
        st.markdown("---")
        with st.expander("ℹ️ About this classifier"):
            st.markdown("""
            This classifier uses a **ResNet50** neural network trained to distinguish between:
            - 🦅 **Birds**: Various bird species
            - ✈️ **Planes**: Aircraft and airplanes
            - 🦸 **Superman**: The iconic superhero
            - ❓ **Other**: Everything else (or low confidence predictions)
            
            **How it works:**
            1. The image is preprocessed and resized to 224x224 pixels
            2. The neural network analyzes the image features
            3. Probabilities are calculated for each class
            4. If the highest probability is below the confidence threshold, the image is classified as "other"
            
            **Tips for best results:**
            - Use clear, well-lit images
            - Ensure the subject is clearly visible
            - Avoid heavily cropped or blurry images
            """)
    
    else:
        # Show sample instructions
        st.info("👆 Upload an image to get started!")
        
        # Show example
        st.markdown("---")
        st.subheader("💡 Example Usage")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Bird** 🦅")
            st.caption("Upload a bird photo")
        
        with col2:
            st.markdown("**Plane** ✈️")
            st.caption("Upload an aircraft image")
        
        with col3:
            st.markdown("**Superman** 🦸")
            st.caption("Upload a Superman image")


if __name__ == "__main__":
    main()
