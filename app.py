import streamlit as st
import os
import shutil
from sample import generate_img
from PIL import Image

# Set up the directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#IMAGE_DIR = os.path.join(BASE_DIR, 'sample', 'bird')

def save_text_to_file(text):
    # Save the text input to a file named bird.txt
    with open('bird.txt', 'w') as f:
        f.write(text)

def display_image(image_path):
    # Display the image at the given path
    image_path=os.path.normpath(image_path)
    img=Image.open(image_path)
    img.show()
    st.image(img, caption='Bird Image')

def keep_latest_folders(folder_path):
    # Get a list of all folders in the folder_path directory
    folder_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    # Sort the folders by creation time
    folder_list.sort(key=lambda x: os.path.getctime(x), reverse=True)
    
    # Keep only the latest 10 folders
    latest_folders = folder_list[:10]
    
    # Delete the rest of the folders
    for folder in folder_list[10:]:
        shutil.rmtree(folder)
        
    print(f"{len(folder_list)-10} folders deleted. Latest 10 folders kept.")
def main():
    # Set the page title and favicon
    st.set_page_config(page_title='Major Project', page_icon='logo.png', layout="wide", 
                       initial_sidebar_state="collapsed")

    # Create a navigation bar
    st.sidebar.image('logo.png', width=150)
    st.sidebar.title('Navigation')
    menu = ['Home', 'About GAN', 'About Us','View Images']
    choice = st.sidebar.selectbox('Select an option', menu)

    # Add a background image
    page_bg_img = '''
    <style>
    body {
    background-image: url("bg.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Show the appropriate tab based on the user's choice
    if choice == 'Home':
        st.title('Image Generation From Textual Description Using Transformer based GAN')
        st.text('Enter a sentence about birds:')
        sentence = st.text_input('', key='input_sentence')
        if st.button('Generate Image', key='generate_image'):
            save_text_to_file(sentence)
            imgp = generate_img()
            IMAGE_DIR = os.path.join(BASE_DIR, imgp)
            image_path = os.path.join(IMAGE_DIR, 'Sent001.png')
            display_image(image_path)
            keep_latest_folders(os.path.join(BASE_DIR, 'samples/bird'))

    elif choice == 'About GAN':
        st.title('About GAN')
        st.write('GAN stands for Generative Adversarial Networks. GANs are a type of machine learning model that are capable of generating new data that is similar to the data they were trained on. GANs consist of two main components: a generator and a discriminator. The generator creates new data, while the discriminator tries to distinguish between real and fake data. The two components are trained together, with the generator trying to create data that the discriminator thinks is real, and the discriminator trying to correctly identify which data is real and which is fake.')
        st.write('In this project, we use a GAN to generate images of birds from textual descriptions.')

    elif choice == 'About Us':
        st.title('About Us')
        st.write('My name is Zubair  and I am a computer science student. This is my final year project, and I am excited to share it with you!')
        st.write('Thanks for checking out my project!')

    elif choice == 'View Images':
        st.title('View Images')
        bimg=os.path.join(BASE_DIR, 'samples/bird')
        #bimg=os.path.normpath(bimg)
        # Get a list of all folders inside the IMAGE_DIR
        folders = os.listdir(bimg)
        print(folders)
        num_cols = 4
        col_count = 0
        row = st.container()
        # Loop through each folder and display the image inside it
        for folder in folders:
            # Check if the item is a folder
            with row:
                if os.path.isdir(os.path.join(bimg, folder)):
                    # Get the path of the image inside the folder
                    image_path = os.path.join(bimg, folder)
                    image_path = os.path.join(image_path,'Sent001.png')

                    # Display the image
                    img=Image.open(image_path)
                    st.image(img, caption=folder, width=300)
                    col_count += 1
                    if col_count == num_cols:
                        col_count = 0
                        row = st.container()
if __name__ == '__main__':
    main()
