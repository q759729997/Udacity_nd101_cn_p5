from keras import layers, models, optimizers
from keras import backend as K


class Critic:
    """
    Critic (Value) Model.评论者模型,将（状态、动作）对映射到它们的 Q 值
    """

    def __init__(self, state_size, action_size, learning_rate=0.001):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.model = None
        self.get_action_gradients = None
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        # 对状态进行规范化
        states_normal = layers.BatchNormalization()(states)
        # 对动作进行规范化
        actions_normal = layers.BatchNormalization()(actions)

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states_normal)
        net_states = layers.Dropout(0.1)(net_states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        net_states = layers.Dropout(0.1)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions_normal)
        net_actions = layers.Dropout(0.1)(net_actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        net_actions = layers.Dropout(0.1)(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
