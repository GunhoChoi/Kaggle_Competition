#Digit Recognizer problem from Kaggle
 https://www.kaggle.com/c/digit-recognizer 를 텐서플로우로 풀어보았다. 정확도는 99.214% 더 높게 나올 방법에는 뭐가 있을까?
 
#Training Process
used ReLU, dropout(0.5), tf.nn.softmax_cross_entropy_with_logits, AdamOptimizer(1e-3)


 1. preprocess data (Pixels & Label)

 2. reshape the pixels into 28x28 shape => [-1, 28, 28, 1] 

 3. 1st 3x3 conv. layer with padding="same" => [-1, 28, 28, 1] -> [-1, 28, 28, 32]
 
 4. 2nd 3x3 conv. layer with padding="same" => [-1, 28, 28, 16] -> [-1, 28, 28, 64]

 5. max pooling [28x28 -> 14x14]

 6. 1st 3x3 conv. layer with padding="same" => [-1, 14, 14, 32] -> [-1, 14, 14, 128]

 7. 2nd 3x3 conv. layer with padding="same" => [-1, 14, 14, 64] ->  [-1, 14, 14, 256]

 8. max pooling [14x14 -> 7x7]

 9. Fully connected layer1 [7x7x128(6272) -> 1024]
 
 10. Dropout(0.5)
 
 11.Fully connected layer2 [1024 -> 10]
