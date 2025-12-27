import os
import cv2 # computer vision : load des images 
import numpy as np
import matplotlib.pyplot as plt # visualisation des chiffres
import tensorflow as tf # pour la partie machine learning

mnist = tf.keras.datasets.mnist # ici on load notre data directement. On n'est pas obligé de chercher un fichier csv
(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1) # ici on normalize notre data. En effet, on fait une mise à l'échelle puisque nos pixels sont entre 0 et 255 et on veut que ça soit entre 0 et 1


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # ici on rend une image de 28x28 pixels en une ligne de 784 pixels
model.add(tf.keras.layers.Dense(128,activation='relu')) # premiere couche de notre neural network avec 128 neuronnes relu permets que Seuls les motifs utiles passent à la couche suivante.
model.add(tf.keras.layers.Dense(128,activation='relu')) # deuxième couche de notre neural network
model.add(tf.keras.layers.Dense(10,activation='softmax')) # dernière couche de notre neural network avec le output et Softmax transforme les scores de sortie du modèle en probabilités normalisées dont la somme vaut 1, afin de choisir la classe la plus probable.

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy']) # ici on compile notre modele. Adam apprend, cross-entropy corrige, accuracy mesure.

model.fit(x_train,y_train, epochs=3)

model.save('handwritten.keras')

model = tf.keras.models.load_model('handwritten.keras')

loss,accuracy = model.evaluate(x_test,y_test)
 
print(loss)
print(accuracy)

# on teste notre modele avec des chiffres qu'on a dessiné avec paint
image_number = 1
while os.path.isfile(f"digits/{image_number}_hand.png"):
    try:
        img = cv2.imread(f"digits/{image_number}_hand.png")[:,:,0]
        img = np.invert(np.array([img]))
        # on load l'image et la met dans le bon format
        prediction = model.predict(img) # on récupère notre prédiction
        print(f"Ce chiffre est probablement {np.argmax(prediction)}") # puisque prediction est un tableau de proba on prend la proba  la plus elevé et on renvoie son indice qui correspond au chiffre qu'on cherche
        plt.imshow(img[0],cmap=plt.cm.binary) # on affiche l'image
        plt.show()
    except:
        print("Error!")
    finally:
        image_number+=1