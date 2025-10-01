import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers, models

def main():
    print("Staring MNIST classification")
    
    print("Loading dataset")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    print("Building simple neural network model")
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Model compiled")
    print("Starting training")
    
    model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=1)
    
    print("Training complete")
    
    print("Evaluating model on test data")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    
    print(f'\n=== FINAL RESULTS ===')
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    print("\n=== SAMPLE PREDICTIONS ===")
    predictions = model.predict(x_test[:5])
    predicted_classes = tf.argmax(predictions, axis=1)
    actual_classes = y_test[:5]
    
    for i in range(5):
        print(f"Sample {i+1}: Predicted={predicted_classes[i].numpy()}, Actual={actual_classes[i]}")
    
if __name__ == "__main__":
    main()
