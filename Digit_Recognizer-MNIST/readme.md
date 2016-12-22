Digit Recognizer problem from Kaggle(https://www.kaggle.com/c/digit-recognizer)

#Training Process

 1. preprocess data (Pixels & Label)

 2. reshape the pixels into 28x28 shape -> [?,28,28,1] 

 3. 1st 3x3 conv. layer with padding="same" -> [28x28 -> 28x28], feature maps 1 -> 16
 
 4. 2nd 3x3 conv. layer with padding="same" -> [28x28 -> 28x28], feature maps 16 -> 32

 5. max pooling [28x28 -> 14x14]

 6. 1st 3x3 conv. layer with padding="same" -> [14x14 -> 14x14], feature maps 32 -> 64

 7. 2nd 3x3 conv. layer with padding="same" -> [14x14 -> 14x14], feature maps 64 -> 128

 8. max pooling [14x14 -> 7x7]

 9. Fully connected layer1 [7x7x128(6272) -> 1024]

 10.Fully connected layer2 [1024 -> 10]

used ReLU, dropout(0.5), AdamOptimizer(1e-3)
