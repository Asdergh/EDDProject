import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    Flatten,
    Dense,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras import (
    Model,
    Sequential
)
from modules import (
    Conv,
    Se
)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import (
    Mean, Accuracy
)
from tensorflow import (
    GradientTape,
    function
)


class SeConvNet(Model):

    def __init__(
            self, 
            input_sh: tuple,
            **kwargs
    ) -> None:
    
        super().__init__(**kwargs)
        self.input_sh = input_sh
        self.model_ = self.build_model_()
    
    def build_model_(self):

        input = Input(shape=self.input_sh)
        conv = Conv(out_channels=128)(input)
        conv = Se(in_channels=128, name="conv1")(conv)
        conv = Conv(out_channels=64)(conv)
        conv = Se(in_channels=64, name="conv2")(conv)
        conv = Conv(out_channels=32)(conv)
        conv = Se(in_channels=32, name="conv3")(conv)

        ffn = Flatten()(conv)
        ffn = Dense(units=128, activation="relu")(ffn)
        ffn = Dense(units=64, activation="relu")(ffn)
        ffn = Dense(units=1, activation="sigmoid")(ffn)

        return Model(inputs=input, outputs=ffn)
    
    def compile(self, optimizer, loss):

        super().compile()
        self.opt = optimizer
        self.loss_fn = loss
        
        self.loss_tracker = Mean(name="loss tracker")
        self.acc_tracker = Accuracy(name="acc tracker")
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.acc_tracker
        ]

    def train_step(self, inputs):

        images, labels = inputs
        with GradientTape() as gr_tape:

            pred_labels = self.model_(images)
            loss = self.loss_fn(labels, pred_labels)
        
        grads = gr_tape.gradient(loss, self.model_.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model_.trainable_variables))
        
        for metric in self.metrics:

            if "loss" in metric.name:
                metric.update_state(loss)
            
            else:
                metric.update_state(labels, pred_labels)
        
        return {
            metric.name: metric.result()
            for metric in self.metrics
        }
    
    @function
    def call(self, inputs):
        return self.total_model_(inputs)

if __name__ == "__main__":

    test_data = np.random.normal(0, 1.0, (100, 128, 128, 3))
    test_labels = np.random.randint(0, 7, 100)
    
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import BinaryCrossentropy
    model = SeConvNet(input_sh=(128, 128, 3))
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=BinaryCrossentropy()
    )

    # model_history = model.fit(
    #     test_data,
    #     test_labels,
    #     batch_size=32,
    #     epochs=10
    # )

    results = model.train_step(inputs=(test_data[:10], np.expand_dims(test_labels[:10], axis=-1)))

    # conv_out = Conv(out_channels=128)(test_data)
    # se_out = Se(in_channels=128)(conv_out)
    # print(se_out.shape)