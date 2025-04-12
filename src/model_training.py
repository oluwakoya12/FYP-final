import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.attention = Dense(1, activation='tanh')

    def call(self, inputs):
        attention_scores = self.attention(inputs)
        attention_scores = tf.squeeze(attention_scores, axis=-1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context_vector = tf.expand_dims(attention_weights, axis=-1) * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


def build_bigru_attention_model(vocab_size, embedding_dim, max_length, gru_units=64, dense_units=64, dropout_rate=0.3,
                                l2_lambda=0.001):
    """Build and compile the BiGRU model with attention"""
    inputs = Input(shape=(max_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, mask_zero=True)(
        inputs)
    gru_out = Bidirectional(GRU(units=gru_units, return_sequences=True, kernel_regularizer=l2(l2_lambda)))(embedding)
    gru_out = Dropout(dropout_rate)(gru_out)
    context_vector, _ = AttentionLayer(units=gru_units * 2)(gru_out)
    dense = Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_lambda))(context_vector)
    dense = Dropout(dropout_rate)(dense)
    outputs = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=64):
    """Train the model with callbacks"""
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.1, patience=2, monitor='val_loss')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    return model, history