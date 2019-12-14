
class Modelfn():

    def __init__(self, hparams):
        self._hparams = hparams

    def _gate_dispatch(self, spikes):
        raise NotImplementedError

    def _gate_combine(self, spikes):
        raise NotImplementedError

    def _tokenizer_network(self, spikes):
        raise NotImplementedError

    def _synthetic_network(self, tokenized_spikes):
        raise NotImplementedError

    def _embedding_network(self, tokenized_spikes, downstream_spikes):
        raise NotImplementedError

    def _target_network(self, embedding_spikes):
        raise NotImplementedError

    def _target_loss(self, embedding_spikes):
        raise NotImplementedError

    def _synthetic_loss(self, embedding_spikes):
        raise NotImplementedError

    def _model_fn(self):

        # Spikes: inputs from the dataset of arbitrary batch_size.
        self.spikes = tf.compat.v1.placeholder(tf.string, [None, 1], name='spikes')

        # Parent gradients: Gradients passed by this components parent.
        self.parent_error = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_embedding], name='parent_grads')

        # Targets: Supervised signals used during training and testing.
        self.targets = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_targets], name='targets')

        # Use Synthetic: Flag, use synthetic inputs when running graph.
        self.use_synthetic = tf.compat.v1.placeholder(tf.bool, shape=[], name='use_synthetic')

        # Gating network.
        with tf.compat.v1.variable_scope("gating_network"):
            gated_spikes = self._gate_dispatch(self.spikes)
            child_inputs = []
            for i, gated_spikes in enumerate(gated_spikes):
                child_inputs.append(_child_input_for_gated_spikes(gated_spikes))
            child_spikes = self._gate_combine(child_inputs)
            gating_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gating_network")

        # Tokenizer network.
        with tf.compat.v1.variable_scope("tokenizer_network"):
            tokenized_spikes = self._tokenizer(self.spikes)
            tokenizer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="tokenizer_network")

        # Synthetic network.
        with tf.compat.v1.variable_scope("synthetic_network"):
            synthetic_spikes = self._synthetic_network(tokenized_spikes)
            synthetic_loss = self._synthetic_loss(synthetic_spikes, self.child_spikes)
            synthetic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="synthetic_network")

        # Downstream switch
        downstream_spikes = tf.cond(
            tf.equal(self.use_synthetic, tf.constant(True)),
            true_fn=lambda: synthetic_spikes,
            false_fn=lambda: child_spikes)

        # Embedding network.
        with tf.compat.v1.variable_scope("embedding_network"):
            self.embedding = self._embedding_network(tokenized_spikes, downstream_spikes)
            embedding_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="embedding_network")

        # Target network
        with tf.compat.v1.variable_scope("target_network"):
            logits = self._target_network(embedding_spikes)
            target_loss = self._target_loss(logits, self.targets)
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")


        # Optimizer
        optimizer = self._optimizer()

        # Synthetic grads.
        synthetic_grads = optimizer.compute_gradients(  loss = synthetic_loss,
                                                        var_list = synthetic_vars)

        # Parent grads
        parent_grads = optimizer.compute_gradients(    loss = self.embedding,
                                                       var_list = embedding_vars,
                                                       grad_loss = self.parent_error)

        # Target grads
        target_grads = optimizer.compute_gradients(    loss = target_loss,
                                                       var_list = target_vars + embedding_vars + gate_vars)

        # Child grads
        child_grads = optimizer.compute_gradients(  loss = target_loss,
                                                    var_list = child_inputs)

        # Synthetic step.
        synthetic_step = optimizer.apply_gradients(synthetic_grads)

        # Parent step.
        parent_step = optimizer.apply_gradients(parent_grads)

        # Target step.
        target_step = optimizer.apply_gradients(parent_grads)
