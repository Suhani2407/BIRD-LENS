import tensorflow as tf
import numpy as np

import wikipediaapi
import wikipedia


class BirdInfo:
    def __init__(self):
        # Load the model
        self.model = tf.keras.models.load_model('static/bird_classification_model_20231023-015251.h5')
        self.class_labels = ['ABBOTS BABBLER', 'ABBOTTS BOOBY', 'ABYSSNIAN GROUND HORNBILL','AFRICAN CROWNED CRANE','AFRICAN EMERALD CUCKOO','AFRICAN FIREFINCH','AFRICAN OYSTER CATCHER','AFRICAN PIED HORNBILL','AFRICAN PYGMY GOOSE','ALBATROSS','ALBERTS TOWHEE','ALEXANDRINE PARAKEET','ALPINE CHOUGH','ALTAMIRA YELLOWTHROAT','AMERICAN AVOCET','AMERICAN BITTERN','AMERICAN COOT','AMERICAN FLAMINGO','AMERICAN GOLDFINCH','AMERICAN KESTREL']

    def get_input(self, path):
        # Preprocess the image
        self.image = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
        self.image = tf.keras.preprocessing.image.img_to_array(self.image)
        self.image = tf.expand_dims(self.image, axis=0)

    def get_prediction(self):
        # Make a prediction
        prediction = self.model.predict(self.image)
        
        # Get the predicted class
        predicted_class = np.argmax(prediction)

        # Get the class label from the predicted class index
        predicted_class_label = self.class_labels[predicted_class]

        return predicted_class_label
    
    def get_wiki(self, name):
        # Function to fetch information from Wikipedia
        in_search = wikipedia.search(name, results = 1)
        input_string = in_search[0]

        wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='Temp (temp@example.com)'
        )

        page = wiki_wiki.page(input_string)
        if page.exists():
            return {'title': page.title, 'summary': page.summary, 'src': page.fullurl}
        else:
            return f'Additional Info not Found!'
