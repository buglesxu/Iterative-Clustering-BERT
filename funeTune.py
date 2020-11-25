from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
# from bert4keras.snippets import open
from keras.layers import Dropout, Dense
import pickle

# 基本信息
maxlen = 128
epochs = 13  # 13
batch_size = 16  # 32
learning_rate = 4e-5
crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率

config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, num) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            labels = [0] * len(token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([num])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []





def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]

        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                right += 1
        total += len(y_true)
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(vaild_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model2.weights')
        print(val_acc)
        print(self.best_val_acc)


if __name__ == "__main__":
    output = open('./qinghua/train_texts_0.01.pkl', 'rb')
    train_texts = pickle.load(output)
    output = open('./qinghua/dev_texts_0.01.pkl', 'rb')
    dev_texts = pickle.load(output)
    output = open('./qinghua/train_ids_0.01.pkl', 'rb')
    train_ids = pickle.load(output)
    output = open('./qinghua/dev_ids_0.01.pkl', 'rb')
    dev_ids = pickle.load(output)

    classify_num_labels = 8

    train_data = []
    for i in range(len(train_ids)):
        train_data.append((train_texts[i], train_ids[i]))
    vaild_data = []
    for i in range(len(dev_ids)):
        vaild_data.append((dev_texts[i], dev_ids[i]))

    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    bert = build_transformer_model(

        config_path=config_path,

        checkpoint_path=checkpoint_path,

        with_pool=True,

        return_keras_model=False,

    )

    classify_output = Dropout(rate=0.1)(bert.model.output)
    classify_output = Dense(units=classify_num_labels,
                            activation='softmax',
                            name='classify_output',
                            kernel_initializer=bert.initializer
                            )(classify_output)

    model = keras.models.Model(bert.model.input, classify_output)
    model.summary()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate),
        metrics=['accuracy'],

    )


    train_generator = data_generator(train_data, batch_size)
    vaild_generator = data_generator(vaild_data, batch_size)
    evaluator = Evaluator()

    # adversarial_training(model, 'Embedding-Token', 0.2)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        # class_weight = 'auto',
        callbacks=[evaluator]
    )