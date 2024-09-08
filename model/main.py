import numpy as np  
import tensorflow as tf  
from tensorflow.keras.layers import Input, RNN, Dense, Embedding, Attention  
from tensorflow.keras.models import Model  
from tensorflow.keras.optimizers import Adam 
# Instance parameter settings  
num_locations = 20    
truck_capacity = 100  
drone_capacity = 10  
time_windows = np.random.randint(8:00, 17:00)  
obstacles = np.random.randint(0, 100) 
# Pointer Mechanism Layer  
class PointerLayer(tf.keras.layers.Layer):  
    def __init__(self, units, **kwargs):  
        super(PointerLayer, self).__init__(**kwargs)  
        self.dense = Dense(units=units, activation='softmax')  
    def call(self, inputs, encoder_outputs):  
 # calculate attention scores  
      attention_scores=tf.matmul(inputs, encoder_outputs, transpose_b=True)  
      attention_weights = self.dense(attention_scores)  
      return attention_weights 
# modeling   
def build_pointer_network(input_dim, hidden_dim, output_dim):  
    encoder_inputs = Input(shape=(None,), dtype='int32')  
    encoder_embeddings=Embedding(input_dim=input_dim, output_dim=hidden_dim)(encoder_inputs)  
    encoder_outputs, encoder_state=RNN(hidden_dim, return_state=True) (encoder_embeddings)  
    decoder_inputs = Input(shape=(None,), dtype='int32')  
    decoder_embeddings=Embedding(input_dim=input_dim, output_dim=hidden_dim)(decoder_inputs)  
   decoder_rnn=RNN(hidden_dim, return_sequences=True, return_ state =True)  
   decoder_outputs, _, _ =decoder_rnn(decoder_embeddings, initial_state =encoder_state)  
   pointer_layer = PointerLayer(units=output_dim)  
   attention_weights = pointer_layer(decoder_outputs, encoder_outputs)  
   model = Model([encoder_inputs, decoder_inputs], attention_weights)  
return model  
model = build_pointer_network(num_locations, 128, num_locations)  
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy')  
   encoder_input_data = np.random.randint(0, num_locations, size=(1, 10))  
decoder_input_data = np.arange(10) + 1 
def decode_sequence(model, encoder_input_data, max_length):  
   encoder_states = model.layers[1].output  # encoder_states is the hidden state of  
  encoder_output = model.layers[2](encoder_states)  # obtain the final output of the encoder
  
    # initialize decoder status
def decode_sequence(model, encoder_input_data, max_length, start_token= None, end_token=None):  
    # initialize the input of the decoder  
 if start_token is None:  
    # select the first point as the starting point  
   decoder_input = np.array([[0]])  # assuming 0 is the index of the start point  
else:  
   decoder_input = np.array([[start_token]])  
   decoded_sequence = [start_token] if start_token is not None else []  
    # decoding process  
for _ in range(max_length):  
        predicted_probs=model.predict([encoder_input_data, decoder_input])  
        next_index = np.argmax(predicted_probs[0, -1, :])  # select the index with the highest attention weight in the last time step 
        # add the selected point to the decoding sequence  
       decoded_sequence.append(next_index)  
        # prepare for the next decoder input  
       decoder_input = np.array([[next_index]])  
        # check if the end conditions have been met  
 if next_index == end_token or len(decoded_sequence) >= max_length:  
            break  
return decoded_sequence  
     encoder_input_data = np.random.randint(0, num_locations, size=(1, 10)) 
     # decoding sequence  
decoded_sequence = decode_sequence(model, encoder_input_data, max_length=num_locations)  
print("Decoded Sequence:", decoded_sequence)  
