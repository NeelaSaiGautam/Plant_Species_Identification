import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
import csv



model = load_model('./plantss.h5', compile=False)
lab = {0: 'Anemone flower', 1: 'Asparagus', 2: 'Banana', 3: 'Chamomile', 4: 'Chickweed', 5: 'Daffodil flower', 6: 'Daisy Fleabane', 7: 'Echinacea', 8: 'Hyacinth flower', 9: 'Jonquil flower', 10: 'Lisianthus flower', 11: 'Madagascar Periwinkle', 12: 'Mini Carnation purple', 13: 'Mustard', 14: 'Pickerelweed flower', 15: 'Poinsettia flower', 16: 'Pompon flower', 17: 'Primrose blue', 18: 'Protea', 19: 'Purple Deadnettle flower', 20: 'Ranunculus flower', 21: 'Rose hips', 22: 'Trachelium flower', 23: 'Vervain Mallow flower', 24: 'Waxflower', 25: 'Wild Grape Vine', 26: 'Wild Leek', 27: 'Aeonium', 28: 'Agapanthus', 29: 'Almond', 30: 'Aloe vera', 31: 'Amaryllis flower', 32: 'Aster', 33: 'Baby breath flower', 34: 'Black rose flower', 35: 'Blue chicory', 36: 'Blue vervain', 37: 'Bougainvillea flower', 38: 'Bromeliad', 39: 'Buttercup flower', 40: 'Calendula flower', 41: 'Canna', 42: 'Cannabis flower', 43: 'Carex', 44: 'cattails', 45: 'Coconut', 46: 'Cone flower', 47: 'Coronation gold', 48: 'Crimson clover', 49: 'Crocus blue', 50: 'Daisy', 51: 'Dandelion', 52: 'Datura flower', 53: 'Delonix regia flower', 54: 'Downy yellow violet', 55: 'Elderberry flower', 56: 'Fireweed flower', 57: 'Forget me not', 58: 'Golden champa flower', 59: 'Harebell flower', 60: 'Hibiscus flower', 61: 'Iris flower', 62: 'Jasmine flower', 63: 'Joe pye weed', 64: 'Knapweed', 65: 'Larkspur flower', 66: 'Lily flower', 67: 'Lotus flower', 68: 'Mallow flower', 69: 'Marigold flower', 70: 'Milk thistle flower', 71: 'Mullein flower yellow', 72: 'Mushroom', 73: 'Narcissistic flower', 74: 'Oleander flower', 75: 'Orchid flower', 76: 'Palash flower', 77: 'Parlor palm', 78: 'Prickly pear cactus', 79: 'Queen anne s lace flower', 80: 'Red hot poker', 81: 'Red clover', 82: 'Red rose', 83: 'Saffron flower', 84: 'Sedum purple', 85: 'Solidago flower', 86: 'St john', 87: 'Statice flower', 88: 'Sunflower', 89: 'Teasel flower', 90: 'Touch me not flower', 91: 'Tuberose flower', 92: 'Tulip flower', 93: 'Viola flower', 94: 'White yarrow', 95: 'Wild bee balm flowers', 96: 'Yellow sow thistle', 97: 'Yucca', 98: 'Zinnia flower red'}


def processed_img(img_path):
    img = load_img(img_path, target_size=(64, 64, 3))
    img = img_to_array(img)
    img = img/255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res


def run():
    img1 = Image.open('./projectnewlogo.jfif')
    img1 = img1.resize((150, 150))
    st.image(img1, use_column_width=False)
    st.title("Plants Species Identification")
    st.markdown('''<h5 style='text-align:left; color:black;'>* Data is based on 99 plant species<br>* Give coloured pictures only</h5>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("choose an image of plant", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file, use_column_width=False)
        save_image_path = './Plant_Data/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            print(result)
            st.success('Predicted plant is:')
            st.success(result)
            st.success(['name','scientific name','water consumption','Environmental Conditions','Uses'])
            with open('project_final.csv','r',encoding='utf-8') as file_obj:
                reader_obj = csv.reader(file_obj)
                for row in reader_obj:
                    if(row[0]==result):
                        st.success(row)



run()
