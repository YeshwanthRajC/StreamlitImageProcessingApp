import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

# Function to convert an image to a water color sketch
def convert_to_watercolor_sketch(inp_img):
    img_1 = cv2.edgePreservingFilter(inp_img, flags=2, sigma_s=50, sigma_r=0.8)
    img_water_color = cv2.stylization(img_1, sigma_s=100, sigma_r=0.5)
    return img_water_color

# Function to convert an image to a pencil sketch
def pencil_sketch(inp_img):
    img_pencil_sketch, pencil_color_sketch = cv2.pencilSketch(
        inp_img, sigma_s=40, sigma_r=0.05, shade_factor=0.0700
    )
    return img_pencil_sketch

# Improved edge detection function
def feature_detection(inp_img, low_threshold=50, high_threshold=150, use_adaptive_threshold=False):
    gray_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    return edges

def Grayscale(inp_img):
    img = np.array(inp_img)  # Convert to NumPy array
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def Blur_the_Image(inp_img):
    bilateral=cv2.GaussianBlur(inp_img,(9,9),cv2.BORDER_DEFAULT)
    return bilateral

def hsv(inp_img):
    hsv_image=cv2.cvtColor(inp_img,cv2.COLOR_BGR2HSV)
    return hsv_image

def lab(inp_img):
    # Convert the input image to LAB color space
    lab_img = cv2.cvtColor(np.array(inp_img), cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into individual channels
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    
    # Merge the channels back together
    merged_img = cv2.merge([l_channel, a_channel, b_channel])
    
    # Convert the merged LAB image back to RGB for display
    #lab_to_rgb = cv2.cvtColor(merged_img, cv2.COLOR_LAB2RGB)
    return merged_img

# Function to load an image
def load_an_image(image):
    img = Image.open(image)
    return img

# The main function which has the code for the web application
def main():
    # Basic heading and titles
    st.title('Convert Image to Any Form')
    st.subheader("Please Upload Your Image")
    
    # Image file uploader
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

    # If the image is uploaded then execute these lines of code
    if image_file is not None:
        # Select box (drop down to choose between water color / pencil sketch)
        option = st.selectbox(
            'How would you like to convert the image',
            ('Convert to water color sketch', 'Convert to pencil sketch', 'Feature Extraction', 'Convert to grayscale Image','Blur Image','HSV MODIFICATION','Lab Color Space Revisited')
        )
        
        if option == 'Convert to water color sketch':
            image = Image.open(image_file)
            final_sketch = convert_to_watercolor_sketch(np.array(image))
            im_pil = Image.fromarray(final_sketch)
            
            # Two columns to display the original image and the image after applying water color sketching effect
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original Image")
                st.image(load_an_image(image_file), width=250)
                
            with col2:
                st.header("Water Color Sketch")
                st.image(im_pil, width=250)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="watercolorsketch.png",
                    mime="image/png"
                )
        
        if option == 'Convert to pencil sketch':
            image = Image.open(image_file)
            final_sketch = pencil_sketch(np.array(image))
            im_pil = Image.fromarray(final_sketch)
            
            # Two columns to display the original image and the image after applying pencil sketching effect
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original Image")
                st.image(load_an_image(image_file), width=250)
                
            with col2:
                st.header("Pencil Sketch")
                st.image(im_pil, width=250)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="pencilsketch.png",
                    mime="image/png"
                )
        
        if option == 'Feature Extraction':
            image = Image.open(image_file)
            st.sidebar.header("Feature Detection Parameters")
            low_threshold = st.sidebar.slider("Low Threshold", 0, 255, 50)
            high_threshold = st.sidebar.slider("High Threshold", 0, 255, 150)
            use_adaptive_threshold = st.sidebar.checkbox("Use Adaptive Threshold", False)
            final_sketch = feature_detection(np.array(image), low_threshold, high_threshold, use_adaptive_threshold)
            im_pil = Image.fromarray(final_sketch)
            
            # Two columns to display the original image and the image after applying edge detection effect
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original Image")
                st.image(load_an_image(image_file), width=250)
                
            with col2:
                st.header("Feature Detected Image")
                st.image(im_pil, width=250)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="featuredetected.png",
                    mime="image/png"
                )
        
        if option == 'Convert to grayscale Image':
            image = Image.open(image_file)
            final_sketch = Grayscale(image)
            im_pil = Image.fromarray(final_sketch).convert("L")  # Ensure the image is in grayscale mode
            
            # Two columns to display the original image and the image after applying grayscale conversion
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original Image")
                st.image(load_an_image(image_file), width=250)
                
            with col2:
                st.header("Grayscale Image")
                st.image(im_pil, width=250)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="grayscale.png",
                    mime="image/png"
                )
        if option == 'Blur Image':
            image = Image.open(image_file)
            final_sketch = Blur_the_Image(np.array(image))
            im_pil = Image.fromarray(final_sketch)
            
            # Two columns to display the original image and the image after applying water color sketching effect
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original Image")
                st.image(load_an_image(image_file), width=250)
                
            with col2:
                st.header("Blurred Image")
                st.image(im_pil, width=250)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="Blurredimg.png",
                    mime="image/png"
                )
        if option == 'HSV MODIFICATION':
            image = Image.open(image_file)
            final_sketch = hsv(np.array(image))
            im_pil = Image.fromarray(final_sketch)
            
            # Two columns to display the original image and the image after applying pencil sketching effect
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original Image")
                st.image(load_an_image(image_file), width=250)
                
            with col2:
                st.header("Enhanced Image")
                st.image(im_pil, width=250)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="enhanced.png",
                    mime="image/png"
                )

        if option == 'Lab Color Space Revisited':
            image = Image.open(image_file)
            final_sketch = lab(image)
            im_pil = Image.fromarray(final_sketch)
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original Image")
                st.image(load_an_image(image_file), width=250)
                with col2:
                 st.header("LAB Color Space Image")
                 st.image(im_pil, width=250)
                 buf = BytesIO()
                 img = im_pil
                 img.save(buf, format="JPEG")
                 byte_im = buf.getvalue()
                 st.download_button(
                 label="Download image",
                 data=byte_im,
                 file_name="lab.png",
                 mime="image/png"
                 )
if __name__ == '__main__':
    main()
