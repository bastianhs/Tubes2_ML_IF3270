# Tugas Besar 2 IF3270 Pembelajaran Mesin: CNN & RNN

Repository ini berisi implementasi Convolutional Neural Network (CNN) dan Recurrent Neural Network (RNN), termasuk Simple RNN dan Long-Short Term Memory (LSTM), untuk memenuhi Tugas Besar 2 mata kuliah IF3270 Pembelajaran Mesin. Proyek ini mencakup pelatihan model menggunakan Keras, analisis hyperparameter, dan implementasi forward propagation *from scratch*.

## Deskripsi Singkat Repository

Proyek ini bertujuan untuk:
1.  Mengimplementasikan dan melatih model **Convolutional Neural Network (CNN)** untuk klasifikasi gambar menggunakan dataset CIFAR-10.
2.  Mengimplementasikan dan melatih model **Simple Recurrent Neural Network (Simple RNN)** untuk klasifikasi teks sentimen berbahasa Indonesia menggunakan dataset NusaX-Sentiment.
3.  Mengimplementasikan dan melatih model **Long-Short Term Memory (LSTM)** untuk klasifikasi teks sentimen berbahasa Indonesia menggunakan dataset NusaX-Sentiment.
4.  Melakukan analisis pengaruh berbagai hyperparameter terhadap kinerja masing-masing model.
5.  Mengimplementasikan fungsi **forward propagation *from scratch*** untuk arsitektur model yang telah dilatih, menggunakan bobot yang sama hasil pelatihan Keras.
6.  Membandingkan hasil prediksi dari implementasi *from scratch* dengan hasil prediksi dari Keras menggunakan metrik **macro f1-score**.

## Fitur Utama

* **CNN untuk Klasifikasi Gambar (CIFAR-10):**
    * Pelatihan model dengan Keras.
    * Analisis pengaruh:
        * Jumlah layer konvolusi.
        * Banyak filter per layer konvolusi.
        * Ukuran filter per layer konvolusi.
        * Jenis pooling layer (Max Pooling vs Average Pooling).
    * Implementasi forward propagation *from scratch* untuk layer Conv2D, Pooling, Flatten, GlobalPooling, dan Dense.
    * Penggunaan Sparse Categorical Crossentropy sebagai loss function dan Adam sebagai optimizer.
    * Pembagian dataset menjadi train (40k), validation (10k), dan test (10k).
* **Simple RNN untuk Klasifikasi Teks (NusaX-Sentiment):**
    * Preprocessing teks: Tokenization menggunakan `TextVectorization` Keras dan Embedding menggunakan `Embedding Layer` Keras.
    * Pelatihan model dengan Keras.
    * Analisis pengaruh:
        * Jumlah layer RNN.
        * Banyak cell RNN per layer.
        * Jenis layer RNN berdasarkan arah (unidirectional vs bidirectional).
    * Implementasi forward propagation *from scratch* untuk layer Embedding, Simple RNN, dan Bidirectional RNN.
    * Penggunaan Sparse Categorical Crossentropy sebagai loss function dan Adam sebagai optimizer.
* **LSTM untuk Klasifikasi Teks (NusaX-Sentiment):**
    * Preprocessing teks: Tokenization menggunakan `TextVectorization` Keras dan Embedding menggunakan `Embedding Layer` Keras.
    * Pelatihan model dengan Keras.
    * Analisis pengaruh:
        * Jumlah layer LSTM.
        * Banyak cell LSTM per layer.
        * Jenis layer LSTM berdasarkan arah (unidirectional vs bidirectional).
    * Implementasi forward propagation *from scratch* untuk layer Embedding, LSTM, dan Bidirectional LSTM.
    * Penggunaan Sparse Categorical Crossentropy sebagai loss function dan Adam sebagai optimizer.
* **Implementasi Modular:** Forward propagation diimplementasikan secara modular per layer dengan membuat kelas untuk tiap layer.
* **Evaluasi:** Perbandingan performa menggunakan macro f1-score.

## Struktur Direktori
```
├── src/
│   ├── layers/
│   │   ├── Conv2D.py
│   │   ├── Pooling.py
│   │   ├── GlobalPooling.py
│   │   ├── Flatten.py
│   │   ├── SimpleRNN.py
│   │   ├── LSTM.py
│   │   ├── Bidirectional.py
│   │   ├── Embedding.py
│   │   └── Dense.py
│   ├── model/
│   │   └── Model.py
│   └── utils/
│       └── ActivationFunction.py
├── notebook
│   ├── CNN/
│   │   ├── Testing_1.ipynb
│   │   ├── Testing_2.ipynb
│   │   ├── Testing_3.ipynb
│   │   └── Testing_4.ipynb
│   ├── RNN/
│   │   ├── Testing_1.ipynb
│   │   ├── Testing_2.ipynb
│   │   └── Testing_3.ipynb
│   ├── LSTM/
│   │   ├── Testing_1.ipynb
│   │   ├── Testing_2.ipynb
│   │   └── Testing_4.ipynb
│   └── Scratch/
│       ├── CNN.ipynb
│       ├── RNN.ipynb
│       └── LSTM.ipynb
├── doc/
│   └── Laporan Tugas Besar 2 IF3270 Pembelajaran Mesin - Kelompok 28.pdf
├── datasets/
│   └── indonesian/
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
├── README.md
└── requirements.txt
```

## Teknologi yang Digunakan

* **Python 3.x**
* **NumPy**: Untuk operasi numerik dan implementasi *from scratch*.
* **TensorFlow & Keras**: Untuk membangun, melatih, dan mengevaluasi model deep learning.
* **Scikit-learn**: Untuk perhitungan metrik evaluasi (macro f1-score).
* **Matplotlib**: Untuk visualisasi grafik loss dan hasil analisis.
* **Jupyter Notebook**: Untuk eksperimen, analisis, dan pelaporan interaktif.

## Setup dan Instalasi

1.  **Clone repository ini:**
    ```bash
    git clone https://github.com/bastianhs/Tubes2_ML_IF3270.git
    cd Tubes2_ML_IF3270
    ```
2.  **Install dependensi yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Dataset:**
    * **CIFAR-10**: Akan diunduh secara otomatis oleh Keras saat pertama kali dijalankan.
    * **NusaX-Sentiment**: Pastikan dataset tersedia dan berada di folder `datasets`.

## Cara Menjalankan Program

1.  **Run the notebook:**   
    Ada beberapa file notebook yang dapat Anda jalankan.  
    Cukup tekan tombol play.  

## Pembagian tugas

<table>
    <tr>
        <th style="text-align: center;">NIM</th>
        <th style="text-align: center;">Nama</th>
        <th style="text-align: center;">Tugas</th>
    </tr>
    <tr>
        <td>13522006</td>
        <td>Agil Fadillah Sabri</td>
        <td>
            Implementasi kelas Layer (Conv2D,  Pooling, Flatten, GlobalPooling, Bidirectional, LSTM, Dense) dan Model; pengujian hyperparameter; pembuatan laporan.
        </td>
    </tr>
    <tr>
        <td>13522034</td>
        <td>Bastian H Suryapratama</td>
        <td>
            Implementasi kelas Layer (SimpleRNN); pengujian hyperparameter; pembuatan laporan.
        </td>
    </tr>
    <tr>
        <td>13522052</td>
        <td>Haikal Assyauqi</td>
        <td>
            Implementasi kelas Layer (Embedding); pengujian hyperparameter; pembuatan laporan.
        </td>
    </tr>
</table>






