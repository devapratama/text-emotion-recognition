import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import pandas as pd
import openpyxl
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Definisi lapisan custom dengan dekorator untuk serialisasi
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

@tf.keras.utils.register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        })
        return config

# Fungsi untuk memuat model dengan lapisan custom
def load_model(model_path):
    custom_objects = {
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Function to load tokenizer and label encoder with st.cache_data
@st.cache_data
def load_support_files(tokenizer_path, label_encoder_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(label_encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)
    return tokenizer, label_encoder

# Fungsi prediksi emosi
def predict_emotion(text, model, tokenizer, label_encoder):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=40)
    prediction = model.predict(padded_sequence)
    label_index = np.argmax(prediction, axis=1)[0]
    label = label_encoder.inverse_transform([label_index])
    return label[0]

def predict_bulk(model, tokenizer, label_encoder, data):
    sequences = tokenizer.texts_to_sequences(data['Review'].tolist()) 
    padded_sequences = pad_sequences(sequences, maxlen=40)
    predictions = model.predict(padded_sequences)
    prediction_indices = np.argmax(predictions, axis=1)
    prediction_labels = label_encoder.inverse_transform(prediction_indices)  # Mengubah indeks numerik menjadi label kelas
    return prediction_labels

# Fungsi untuk mengkonversi DataFrame ke CSV
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.read()

def read_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        st.error("This file format is not supported! Please upload a CSV or Excel file.")
        return None
    
def create_sample_file(file_format):
    sample_data = {'Review': [
        'Saya akan merekomendasikan ini kepada teman.',
        'Makanan tidak enak dan terlalu mahal.',
        'Sangat mudah digunakan dan diatur.',
        'Saya merasa tertipu dengan produk ini.',
        'Kamera ini memiliki fitur yang hebat.',
        'Akan membeli lagi dari penjual ini.',
        'Pengembalian dana yang mudah dan layanan hebat.',
        'Pengiriman cepat, terima kasih!',
        'Pelayanan yang ramah dan responsif.',
        'Kualitas suara dari headphone ini luar biasa.',
        'Harga terjangkau dan kualitas bagus.',
        'Produk ini benar-benar memuaskan.',
        'Tidak bernilai uang yang saya bayar.',
        'Buku ini sangat membosankan dan tidak informatif.',
        'Pengiriman terlambat dan paket rusak.',
        'Instruksi yang disertakan tidak jelas.',
        'Tidak sesuai dengan deskripsi.',
        'Sangat kecewa dengan pembelian ini.',
        'Produk ini lebih dari yang saya harapkan.'
    ]}
    df_sample = pd.DataFrame(sample_data)

    if file_format == 'csv':
        return df_sample.to_csv(index=False).encode('utf-8')
    elif file_format == 'excel':
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_sample.to_excel(writer, index=False)
        output.seek(0)
        return output.getvalue()
        
# Get the directory where the script is located
base_path = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model and support files
model_path = os.path.join(base_path, 'transformer_emotion.keras')
tokenizer_path = os.path.join(base_path, 'tokenizer.pickle')
label_encoder_path = os.path.join(base_path, 'label_encoder.pickle')

model = load_model(model_path)
tokenizer, label_encoder = load_support_files(tokenizer_path, label_encoder_path)

tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction"])

with tab1:
    st.title('Prediksi Emosi dari Teks Review')
    user_input = st.text_area("Masukkan teks review di sini:")
    if st.button('Prediksi'):
        if user_input:
            predicted_emotion = predict_emotion(user_input, model, tokenizer, label_encoder)
            st.write(f'Emosi yang diprediksi: **{predicted_emotion}**')
        else:
            st.write('Silakan masukkan teks untuk prediksi.')

with tab2:
    st.write("Upload a CSV or Excel file for multi predictions.")
    
    with st.container():
        col1, col2 = st.columns([1, 2.7])
        with col1:
            st.download_button(
                label="Download Sample CSV",
                data=create_sample_file('csv'),
                file_name="sample_input.csv",
                mime="text/csv",
                key="sample-csv"
            )
        with col2:
            st.download_button(
                label="Download Sample Excel",
                data=create_sample_file('excel'),
                file_name="sample_input.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="sample-excel"
            )

    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            file_type = uploaded_file.type
                
            if file_type == "text/csv":
                try:
                    data = pd.read_csv(
                        uploaded_file,
                        encoding='utf-8',
                        on_bad_lines='skip'
                    )
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
                
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                try:
                    data = pd.read_excel(uploaded_file, engine='openpyxl')
                except Exception as e:
                    st.error(f"Error reading Excel file: {e}")

            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")

            if not data.empty:
                predictions = predict_bulk(model, tokenizer, label_encoder, data)
                predictions_df = pd.DataFrame(predictions, columns=['Emotion Prediction'])
                results = pd.concat([data, predictions_df], axis=1)
                
                st.write("Results with Predictions:")
                st.dataframe(results)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=convert_df_to_csv(results),
                        file_name='predictions.csv',
                        mime='text/csv',
                        key="predictions-csv"
                    )
                with col2:
                    st.download_button(
                        label="Download Predictions as Excel",
                        data=convert_df_to_excel(results),
                        file_name='predictions.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key="predictions-excel"
                    )

                fig, ax = plt.subplots()
                barplot = sns.countplot(x='Emotion Prediction', data=results, hue='Emotion Prediction', palette='viridis', legend=False, ax=ax)
                ax.set_title('Distribution of Predicted Emotions')

                for p in barplot.patches:
                    count = round(p.get_height())
                    barplot.annotate('{}'.format(count), 
                                     (p.get_x() + p.get_width() / 2., count),
                                     ha = 'center', va = 'center', 
                                     xytext = (0, 9), 
                                     textcoords = 'offset points')

                st.pyplot(fig)

                emotion_counts = results['Emotion Prediction'].value_counts()
                fig2, ax2 = plt.subplots()
                ax2.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
                ax2.axis('equal')
                ax2.set_title('Emotion Prediction Distribution')
                st.pyplot(fig2)
            else:
                st.write("File is empty or not properly formatted.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
